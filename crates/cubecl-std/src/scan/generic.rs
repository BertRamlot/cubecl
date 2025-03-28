use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, CubeElement, server::Handle};
use pretty_assertions::assert_eq;

use std::fmt::Display;

const CUBE_SIZE: u32 = 256;
const MIN_SUBGROUP_SIZE: u32 = 4;
const MAX_REDUCE_SIZE: u32 = CUBE_SIZE / MIN_SUBGROUP_SIZE;

const PART_SIZE: u32 = 4096;

// Merrill, Duane, and Michael Garland. "Single-pass parallel prefix scan with decoupled look-back." NVIDIA, Tech. Rep. NVR-2016-002 (2016).
#[cube(launch_unchecked)]
fn prefix_sum_kernel<C: Int>(
    scan_in: &Tensor<Line<C>>,
    scan_out: &mut Tensor<Line<C>>,
    scan_bump: &Tensor<Atomic<C>>,
    reduction: &Tensor<Atomic<C>>, // Expected to be zero (or atleast the flags)
    cube_count_x: u32,
) {
    // Comptime constants
    let line_spt = comptime!(PART_SIZE / CUBE_SIZE / scan_in.line_size());
    let v_last = comptime!(scan_in.line_size() - 1);
    let nums_per_cube = CUBE_SIZE * line_spt;

    let mut partition_broadcast = SharedMemory::<C>::new(1);
    let mut broadcast = SharedMemory::<C>::new(1);
    let mut reduce = SharedMemory::<C>::new(MAX_REDUCE_SIZE);
    let batch = CUBE_POS_Z;

    // Acquire partition index (part_id)
    if UNIT_POS_X == 0 {
        partition_broadcast[0] = Atomic::add(&scan_bump[batch], C::new(1));
    }
    sync_units();
    let part_id = u32::cast_from(partition_broadcast[0]);

    let plane_id = UNIT_POS_X / PLANE_DIM;
    let start_plane = part_id * nums_per_cube + plane_id * PLANE_DIM * line_spt;
    // Exit if the whole plane is out of bounds
    if start_plane >= scan_in.shape(1) {
        terminate!();
    }

    // Define flag values for reduction and scanning
    let flag_reduction = C::new(1); // Indicates the reduction is available
    let flag_inclusive = C::new(2); // Indicates the inclusive sum is available
    let flag_mask = C::new(3);

    let zero = C::new(0);

    // Calculate offsets for reduction and scanning
    let red_offs = batch * reduction.stride(0);
    let scan_offs = batch * scan_in.stride(0);

    let mut t_scan = Array::<Line<C>>::vectorized(line_spt, scan_in.line_size());
    {
        let mut i = start_plane + UNIT_POS_PLANE;

        if part_id < cube_count_x - 1 {
            for k in 0..line_spt {
                // Manually fuse not_equal and cast
                let mut scan = Line::cast_from(scan_in[i + scan_offs].not_equal(Line::new(zero)));
                #[unroll]
                for v in 1..scan_in.line_size() {
                    let prev = scan[v - 1];
                    scan[v] += prev;
                }
                t_scan[k] = scan;
                i += PLANE_DIM;
            }
        } else if part_id == cube_count_x - 1 {
            // Last partition might not be full and hence requires special attention
            for k in 0..line_spt {
                if i < scan_in.shape(1) {
                    // Manually fuse not_equal and cast
                    let mut scan =
                        Line::cast_from(scan_in[i + scan_offs].not_equal(Line::new(zero)));
                    #[unroll]
                    for v in 1..scan_in.line_size() {
                        let prev = scan[v - 1];
                        scan[v] += prev;
                    }
                    t_scan[k] = scan;
                }
                i += PLANE_DIM;
            }
        }

        let mut prev = zero;
        let plane_mask = PLANE_DIM - 1;
        let circular_shift = (UNIT_POS_PLANE + plane_mask) & plane_mask;
        for k in 0..line_spt {
            let t = plane_broadcast(plane_inclusive_sum(t_scan[k][v_last]), circular_shift);
            t_scan[k] += Line::cast_from(select(UNIT_POS_PLANE != 0, t, zero) + prev);
            prev += plane_broadcast(t, 0);
        }

        if UNIT_POS_PLANE == 0 {
            reduce[plane_id] = prev;
        }
    }
    sync_units();

    //Non-divergent subgroup agnostic inclusive scan across subgroup reductions
    let lane_log = count_trailing_zeros(PLANE_DIM);
    let spine_size = CUBE_DIM >> lane_log;
    {
        let mut offset_0 = 0;
        let mut offset_1 = 0;
        let aligned_size =
            1 << ((count_trailing_zeros(spine_size) + lane_log + 1) / lane_log * lane_log);
        let mut j = PLANE_DIM;
        while j <= aligned_size {
            let i_0 = ((UNIT_POS_X + offset_0) << offset_1) - offset_0;
            let pred_0 = i_0 < spine_size;
            let t_0 = plane_inclusive_sum(select(pred_0, reduce[i_0], zero));
            if pred_0 {
                reduce[i_0] = t_0;
            }
            sync_units();

            if j != PLANE_DIM {
                let rshift = j >> lane_log;
                let i_1 = UNIT_POS_X + rshift;
                if (i_1 & (j - 1)) >= rshift {
                    let pred_1 = i_1 < spine_size;
                    let t_1 = select(pred_1, reduce[((i_1 >> offset_1) << offset_1) - 1], zero);
                    if pred_1 && ((i_1 + 1) & (rshift - 1)) != 0 {
                        reduce[i_1] += t_1;
                    }
                }
            } else {
                offset_0 += 1;
            }
            offset_1 += lane_log;

            j <<= lane_log;
        }
    }
    sync_units();

    // Store reduction results with flags (Device broadcast)
    if UNIT_POS_X == 0 {
        Atomic::store(
            &reduction[part_id + red_offs],
            (reduce[spine_size - 1] << C::new(2))
                | select(part_id != 0, flag_reduction, flag_inclusive),
        )
    }

    // 4. Determine the partition’s exclusive prefix using decoupled look-back
    if part_id != 0 {
        if UNIT_POS_X == 0 {
            let mut lookback_id = part_id - 1;
            let mut prev_reduction = zero;
            loop {
                let flag_payload = Atomic::load(&reduction[lookback_id + red_offs]);
                if (flag_payload & flag_mask) == flag_inclusive {
                    prev_reduction += flag_payload >> C::new(2);
                    Atomic::store(
                        &reduction[part_id + red_offs],
                        ((prev_reduction + reduce[spine_size - 1]) << C::new(2)) | flag_inclusive,
                    );
                    broadcast[0] = prev_reduction;
                    break;
                }

                if (flag_payload & flag_mask) == flag_reduction {
                    prev_reduction += flag_payload >> C::new(2);
                    lookback_id -= 1;
                }
            }
        }
        sync_units();
    }

    {
        // Final output writing stage
        let prev = if plane_id != 0 {
            reduce[plane_id - 1]
        } else {
            zero
        };
        let prev = Line::cast_from(broadcast[0] + prev);
        let mut i = start_plane + UNIT_POS_PLANE;

        if part_id < cube_count_x - 1 {
            for k in 0..line_spt {
                scan_out[i + scan_offs] = t_scan[k] + prev;
                i += PLANE_DIM;
            }
        } else if part_id == cube_count_x - 1 {
            // Last partition might not be full and hence requires special attention
            for k in 0..line_spt {
                if i < scan_out.shape(1) {
                    scan_out[i + scan_offs] = t_scan[k] + prev;
                }
                i += PLANE_DIM;
            }
        }
    }
}

#[cube]
fn count_trailing_zeros(num: u32) -> u32 {
    u32::find_first_set(num) - 1
}

/// Compute the prefix sum of a tensor
pub fn prefix_sum<R: Runtime, C: Int + CubeElement>(
    client: &cubecl::prelude::ComputeClient<R::Server, R::Channel>,
    len: u32,
    data: &mut Handle, // Shape: [batches, numbers]
) -> Handle {
    let numbers: u32 = len; // TODO: 1 batch assumption
    let batches = len / numbers;

    let input = data; // Shape: [batches, numbers]
    let out = client.empty((batches * numbers) as usize * core::mem::size_of::<C>()); // Shape: [batches, numbers]

    let cubes = numbers.div_ceil(PART_SIZE);
    let cube_dim = CubeDim::new_1d(CUBE_SIZE);
    let cube_count = CubeCount::new_3d(cubes, 1, batches);

    let bump = client.empty((batches) as usize * core::mem::size_of::<C>()); // Shape: [batches,]
    let reduction = client.empty((batches * cubes) as usize * core::mem::size_of::<C>()); // Shape: [batches, cubes]

    unsafe {
        //.as_tensor_arg::<I>(factor),
        prefix_sum_kernel::launch_unchecked::<C, R>(
            client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts::<Line<C>>(
                input,
                &[numbers as usize, 1],
                &[batches as usize, numbers as usize],
                4,
            ),
            TensorArg::from_raw_parts::<Line<C>>(
                &out,
                &[numbers as usize, 1],
                &[batches as usize, numbers as usize],
                4,
            ),
            TensorArg::from_raw_parts::<Atomic<C>>(&bump, &[1], &[batches as usize], 1),
            TensorArg::from_raw_parts::<Atomic<C>>(
                &reduction,
                &[cubes as usize, 1],
                &[batches as usize, cubes as usize],
                1,
            ),
            ScalarArg::new(cubes),
        )
    };

    out
}

// test_identity
pub fn test_prefix_sum<R: Runtime, C: Int + CubeElement + Display>(device: &R::Device, len: u32) {
    let data: Vec<C> = (0..len)
        .map(|_i| C::from_int(1 as i64)) // (7919 * i % 457)
        .collect();

    let client = R::client(device);
    let mut data_handle = client.create(C::as_bytes(&data));
    prefix_sum::<R, C>(&client, len, &mut data_handle);
    let actual_bytes = client.read_one(data_handle.binding());
    let actual = C::from_bytes(&actual_bytes);

    let expected: Vec<C> = data
        .into_iter()
        .scan(C::from_int(0), |x, y| {
            *x += y;
            Some(*x)
        })
        .collect();
    assert_eq!(actual, expected);
    // for (found, reff) in actual.iter().zip(expected) {
    //     assert_eq!(*found, reff);
    // }
}
