#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_scan {
    () => {
        mod scan {
            $crate::testgen_scan!(u32);
        }
    };
    ($numeric:ident) => {
            use super::*;
            use cubecl_linalg::tensor::tests;
            use cubecl_core::flex32;

            pub type NumericT = $numeric;

            #[test]
            pub fn test_tiny() {
                cubecl_std::scan::test_prefix_sum::<TestRuntime, NumericT>(&Default::default(), 5);
            }

            #[test]
            pub fn test_small() {
                cubecl_std::scan::test_prefix_sum::<TestRuntime, NumericT>(&Default::default(), 1000);
            }

            #[test]
            pub fn test_normal() {
                cubecl_std::scan::test_prefix_sum::<TestRuntime, NumericT>(&Default::default(), 100000)
            }

            #[test]
            pub fn test_large() {
                cubecl_std::scan::test_prefix_sum::<TestRuntime, NumericT>(&Default::default(), 10000000)
            }
    };
    ([$($numeric:ident),*]) => {
        mod scan {
            use super::*;
            ::paste::paste! {
                $(mod [<$numeric _ty>] {
                    use super::*;

                    $crate::testgen_scan!($numeric);
                })*
            }
        }
    };
}
