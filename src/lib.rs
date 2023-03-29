#[cfg(test)]
mod tests;

use half::f16;
use std::ffi::{c_int, c_void, CStr, CString};
use std::fmt::Formatter;
use std::mem::size_of;
use std::ptr::{addr_of, addr_of_mut, null_mut};
// use rknn2_sys;
use num_enum::TryFromPrimitive;

use log::warn;

use log::debug;
use rknn2_sys::*;

pub type RkNNResult<T> = Result<T, RkNNError>;

#[derive(Debug)]
pub enum RkNNError {
    InternalError(RkNNInternalError),
    VerificationFailed(&'static str),
    InputCreateFailed,
}

#[derive(Debug, TryFromPrimitive)]
#[repr(i32)]
pub enum RkNNInternalError {
    Fail = RKNN_ERR_FAIL,
    Timeout = RKNN_ERR_TIMEOUT,
    DeviceUnavailable = RKNN_ERR_DEVICE_UNAVAILABLE,
    MallocFail = RKNN_ERR_MALLOC_FAIL,
    ParamInvalid = RKNN_ERR_PARAM_INVALID,
    CtxInvalid = RKNN_ERR_CTX_INVALID,
    InputInvalid = RKNN_ERR_INPUT_INVALID,
    OutputInvalid = RKNN_ERR_OUTPUT_INVALID,
    DeviceUnmatched = RKNN_ERR_DEVICE_UNMATCH,
    IncompatibleOptimizationLevelVersion = RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION,
    TargetPlatformUnmatched = RKNN_ERR_TARGET_PLATFORM_UNMATCH,
    Unknown = 0,
}

impl RkNNInternalError {
    // pub fn result_from_errno<T>(errno: c_int, pending_result: T) -> RkNNResult<T> {
    //     match RkNNError::try_from_primitive(errno) {
    //         Ok(RkNNError::Unknown) => Ok(pending_result),
    //         Ok(error) => Err(error),
    //         Err(_) => Err(RkNNError::Unknown)
    //     }
    // }
    pub fn result_from_errno(errno: c_int) -> RkNNResult<()> {
        match RkNNInternalError::try_from_primitive(errno) {
            Ok(RkNNInternalError::Unknown) => Ok(()),
            Ok(error) => Err(RkNNError::InternalError(error)),
            Err(_) => {
                warn!("Unknown errno: {}", errno);
                Err(RkNNError::InternalError(RkNNInternalError::Unknown))
            }
        }
    }
}

/// Rockchip Neuron Network Context
///
/// This is designed to bind to `rknn_context`.
#[derive(Debug)]
pub struct RkNN(rknn_context);

// impl std::fmt::Debug for RkNN{
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         f.
//     }
// }

impl RkNN {
    // pub fn new_from_slice() -> Self{
    //     rknn2_sys::rknn_init()
    // }
    pub fn new_from_path(path: &str) -> RkNNResult<Self> {
        unsafe {
            let c_path = CString::new(path).expect("CString::new for rknn_init file path failed");
            let mut ctx: rknn_context = 0;
            let errno = rknn_init(&mut ctx, c_path.as_ptr() as *mut c_void, 0, 0, null_mut());
            RkNNInternalError::result_from_errno(errno).map(|_| Self(ctx))
        }
    }

    pub fn set_core_mask(&self, core_mask: CoreMask) -> RkNNResult<()> {
        unsafe {
            let c_core_mask = match core_mask {
                CoreMask::Auto => _rknn_core_mask_RKNN_NPU_CORE_AUTO,
                CoreMask::_0 => _rknn_core_mask_RKNN_NPU_CORE_0,
                CoreMask::_1 => _rknn_core_mask_RKNN_NPU_CORE_1,
                CoreMask::_2 => _rknn_core_mask_RKNN_NPU_CORE_2,
                CoreMask::_0_1 => _rknn_core_mask_RKNN_NPU_CORE_0_1,
                CoreMask::_0_1_2 => _rknn_core_mask_RKNN_NPU_CORE_0_1_2,
            };
            let errno = rknn_set_core_mask(self.0, c_core_mask);
            RkNNInternalError::result_from_errno(errno)
        }
    }

    pub fn dub_context(&mut self) -> RkNNResult<Self> {
        unsafe {
            let mut ctx: rknn_context = 0;
            let errno = rknn_dup_context(&mut self.0, &mut ctx);
            RkNNInternalError::result_from_errno(errno).map(|_| Self(ctx))
        }
    }

    pub fn query_sdk_version(&self) -> RkNNResult<(String, String)> {
        unsafe {
            let mut result = rknn_sdk_version {
                api_version: [0u8; 256],
                drv_version: [0u8; 256],
            };
            let errno = rknn_query(
                self.0,
                _rknn_query_cmd_RKNN_QUERY_SDK_VERSION,
                addr_of_mut!(result) as *mut c_void,
                size_of::<rknn_sdk_version>() as u32,
            );
            RkNNInternalError::result_from_errno(errno).map(|_| {
                (
                    CStr::from_ptr(&result.api_version[0])
                        .to_str()
                        .unwrap()
                        .to_owned(),
                    // String::from_utf8(Vec::from(result.drv_version)).unwrap()
                    // CStr::from_bytes_with_nul_unchecked(&result.drv_version).to_str().unwrap().to_owned()
                    CStr::from_ptr(&result.drv_version[0])
                        .to_str()
                        .unwrap()
                        .to_owned(), // Vec::<u8>::from(result.api_version),
                                     // Vec::<u8>::from(result.drv_version)
                )
            })
        }
    }

    #[allow(non_upper_case_globals)]
    fn get_attr<T: Default + Clone>(
        &self,
        count: u32,
        cmd: _rknn_query_cmd,
        mapper: fn(&mut T) -> &mut TensorAttr,
    ) -> RkNNResult<Vec<T>> {
        const ZEROS: rknn_tensor_attr = rknn_tensor_attr {
            index: 0,
            n_dims: 0,
            dims: [0; 16],
            name: [0; 256],
            n_elems: 0,
            size: 0,
            fmt: 0,
            type_: 0,
            qnt_type: 0,
            fl: 0,
            zp: 0,
            scale: 0.0,
            w_stride: 0,
            size_with_stride: 0,
            pass_through: 0,
            h_stride: 0,
        };

        let mut result = Vec::new();
        result.resize(count as usize, T::default());
        for (index, attr) in result.iter_mut().map(mapper).enumerate() {
            let mut out = rknn_tensor_attr {
                index: index as u32,
                ..ZEROS
            };
            let errno = unsafe {
                rknn_query(
                    self.0,
                    cmd,
                    addr_of_mut!(out) as *mut c_void,
                    size_of::<rknn_tensor_attr>() as u32,
                )
            };
            debug!("{:?}", out);
            // let dims = Box::new(attr.dims[0..(attr.n_dims as usize)]);
            // let dims = out.dims[0..(out.n_dims as usize)].to_vec().into_boxed_slice();
            let dims = match out.fmt {
                _rknn_tensor_format_RKNN_TENSOR_NCHW => TensorDimension::BatchChannelHeightWidth {
                    batch: out.dims[0],
                    channel: out.dims[1],
                    height: out.dims[2],
                    width: out.dims[3],
                },

                _rknn_tensor_format_RKNN_TENSOR_NHWC => TensorDimension::BatchHeightWidthChannel {
                    batch: out.dims[0],
                    height: out.dims[1],
                    width: out.dims[2],
                    channel: out.dims[3],
                },

                _rknn_tensor_format_RKNN_TENSOR_NC1HWC2 => {
                    TensorDimension::BatchChannel1HeightWidthChannel2 {
                        batch: out.dims[0],
                        channel1: out.dims[1],
                        height: out.dims[2],
                        width: out.dims[3],
                        channel2: out.dims[4],
                    }
                }

                _rknn_tensor_format_RKNN_TENSOR_UNDEFINED => TensorDimension::Undefined(
                    out.dims[0..(out.n_dims as usize)]
                        .to_vec()
                        .into_boxed_slice(),
                ),
                _ => todo!("dwandiu"),
            };

            if dims.n_dims() as u32 != out.n_dims {
                Err(RkNNError::VerificationFailed(
                    "n_dims doesn't match dim type",
                ))?
            }
            if dims.multiplication() != out.n_elems {
                Err(RkNNError::VerificationFailed("n_elems doesn't match dims"))?
            }
            // if dims.multiplication() != out.size { Err(RkNNError::VerificationFailed("size unmatched"))? }

            let name = unsafe { CStr::from_ptr(&out.name[0]).to_str().unwrap().to_owned() };

            *attr = TensorAttr {
                index: out.index,
                dims,
                name,
                n_elems: out.n_elems,
                size: out.size,
                // fmt: TensorFormat::try_from(out.fmt).unwrap(),
                data_type: TensorType::try_from(out.fmt).unwrap(),
                qnt_type: match out.qnt_type {
                    _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE => QuantifyType::None,
                    _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP => {
                        QuantifyType::DynamicFixedPoint { fl: out.fl }
                    }
                    _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => {
                        QuantifyType::AffineAsymmetric { scale: out.scale }
                    }
                    _ => todo!("on none"),
                },
                w_stride: out.w_stride,
                size_with_stride: out.size_with_stride,
                pass_through: out.pass_through != 0,
                h_stride: out.h_stride,
            };

            RkNNInternalError::result_from_errno(errno)?;
        }
        Ok(result)
    }
    pub fn query_inputs_outputs(
        &self,
        native: bool,
    ) -> RkNNResult<(Vec<InputTensor>, Vec<OutputTensor>)> {
        let (input_count, output_count) = self.query_in_out_num()?;
        debug!(
            "This .rknn consists of {} inputs and {} outputs",
            input_count, output_count
        );

        let inputs = self.get_attr::<InputTensor>(
            input_count,
            if !native {
                _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR
            } else {
                _rknn_query_cmd_RKNN_QUERY_NATIVE_INPUT_ATTR
            },
            |input_tensor| &mut input_tensor.0,
        )?;
        let outputs = self.get_attr::<OutputTensor>(
            output_count,
            if !native {
                _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR
            } else {
                _rknn_query_cmd_RKNN_QUERY_NATIVE_OUTPUT_ATTR
            },
            |output_tensor| &mut output_tensor.0,
        )?;
        Ok((inputs, outputs))
    }

    pub fn query_in_out_num(&self) -> RkNNResult<(u32, u32)> {
        unsafe {
            let mut result = rknn_input_output_num {
                n_input: 0,
                n_output: 0,
            };
            let errno = rknn_query(
                self.0,
                _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
                addr_of_mut!(result) as *mut c_void,
                size_of::<rknn_input_output_num>() as u32,
            );
            RkNNInternalError::result_from_errno(errno).map(|_| (result.n_input, result.n_output))
        }
    }

    pub fn query_perf_detail(&self) -> RkNNResult<String> {
        unsafe {
            let mut result = rknn_perf_detail {
                perf_data: null_mut(),
                data_len: 0,
            };
            let errno = rknn_query(
                self.0,
                _rknn_query_cmd_RKNN_QUERY_PERF_DETAIL,
                addr_of_mut!(result) as *mut c_void,
                size_of::<rknn_perf_detail>() as u32,
            );
            // TODO: data_len ignored
            RkNNInternalError::result_from_errno(errno).map(|_| {
                CStr::from_ptr(result.perf_data)
                    .to_str()
                    .unwrap()
                    .to_owned()
            })
        }
    }

    pub fn query_custom_string(&self) -> RkNNResult<String> {
        unsafe {
            let mut result = rknn_custom_string { string: [0; 1024] };
            let errno = rknn_query(
                self.0,
                _rknn_query_cmd_RKNN_QUERY_CUSTOM_STRING,
                addr_of_mut!(result) as *mut c_void,
                size_of::<rknn_custom_string>() as u32,
            );
            RkNNInternalError::result_from_errno(errno).map(|_| {
                CStr::from_ptr(&result.string[0])
                    .to_str()
                    .unwrap()
                    .to_owned()
            })
        }
    }

    pub fn inputs_set_c(&self, inputs: &[rknn_input]) -> RkNNResult<()> {
        unsafe {
            let errno = rknn_inputs_set(
                self.0,
                inputs.len() as u32,
                inputs.as_ptr() as *mut _rknn_input,
            );
            RkNNInternalError::result_from_errno(errno)
        }
    }

    /// You should use `bytemuck` to change you typed slice to byte array (TODO)
    pub fn inputs_set(&self, inputs: &[(&InputTensor, &[u8])]) -> RkNNResult<()> {
        unsafe {
            let c_inputs = inputs
                .iter()
                .map(|(InputTensor(attr), data)| {
                    // if attr.size != (size_of::<T>() * data.len()) as u32 {
                    //     warn!("attr.size ({}) != size_of::<T> ({})", attr.size ,size_of::<T>() as u32);
                    // }
                    if attr.size * attr.data_type.byte_size() as u32 != data.len() as u32 {
                        warn!(
                            "attr.size * attr.data_type.byte_size ({}) != data.len ({})",
                            attr.size * attr.data_type.byte_size() as u32,
                            data.len() as u32
                        );
                    }
                    rknn_input {
                        index: attr.index,
                        buf: addr_of!(data[0]) as *mut c_void,
                        size: data.len() as u32,
                        pass_through: attr.pass_through as u8,
                        type_: attr.data_type as u32,
                        fmt: attr.dims.get_fmt_type(),
                    }
                })
                .collect::<Vec<_>>();
            let errno = rknn_inputs_set(
                self.0,
                inputs.len() as u32,
                c_inputs.as_ptr() as *mut _rknn_input,
            );
            RkNNInternalError::result_from_errno(errno)
        }
    }

    pub fn run(&self) -> RkNNResult<()> {
        unsafe {
            let errno = rknn_run(self.0, null_mut());
            RkNNInternalError::result_from_errno(errno)
        }
    }

    pub fn outputs_get_allocated_by_runtime(
        &self,
        outputs: &[OutputTensor],
    ) -> RkNNResult<OutputsMemory> {
        let mut c_outputs = outputs
            .iter()
            .map(|output| rknn_output {
                want_float: u8::from(output.0.data_type == TensorType::Float32),
                is_prealloc: 0,
                index: output.0.index,
                buf: null_mut(),
                size: 0,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let errno = unsafe {
            rknn_outputs_get(
                self.0,
                outputs.len() as u32,
                c_outputs.as_mut_ptr(),
                null_mut(),
            )
        };
        RkNNInternalError::result_from_errno(errno).map(|_| OutputsMemory {
            ctx: self,
            c_outputs,
        })
    }
}

#[derive(Debug)]
pub struct OutputsMemory<'a> {
    ctx: &'a RkNN,
    c_outputs: Box<[rknn_output]>,
}

impl OutputsMemory<'_> {
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        self.c_outputs
            .iter()
            .map(|co| unsafe { std::slice::from_raw_parts(co.buf as *const u8, co.size as usize) })
    }
}

impl Drop for OutputsMemory<'_> {
    fn drop(&mut self) {
        debug!("Releasing outputs");
        unsafe {
            rknn_outputs_release(
                self.ctx.0,
                self.c_outputs.len() as u32,
                addr_of_mut!(self.c_outputs[0]),
            );
        }
    }
}

impl Drop for RkNN {
    fn drop(&mut self) {
        debug!("Destroying RkNN context");
        unsafe {
            rknn_destroy(self.0);
        }
    }
}

pub enum CoreMask {
    Auto,
    _0,
    _1,
    _2,
    _0_1,
    _0_1_2,
    // Undefined = RKNN_NPU_CORE_UNDEFINED,
}

//noinspection SpellCheckingInspection
#[derive(Debug, TryFromPrimitive, Clone, Default)]
#[repr(u32)]
pub enum TensorFormat {
    #[default]
    /// a.k.a. NCHW
    BatchChannelHeightWidth = _rknn_tensor_format_RKNN_TENSOR_NCHW,

    /// a.k.a. NHWC
    BatchHeightWidthChannel = _rknn_tensor_format_RKNN_TENSOR_NHWC,

    /// a.k.a. NC1HWC2
    BatchChannel1HeightWidthChannel2 = _rknn_tensor_format_RKNN_TENSOR_NC1HWC2,

    Undefined = _rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
}

//noinspection SpellCheckingInspection
#[derive(Debug, Clone)]
pub enum TensorDimension {
    /// a.k.a. NCHW
    BatchChannelHeightWidth {
        batch: u32,
        channel: u32,
        height: u32,
        width: u32,
    },

    /// a.k.a. NHWC
    BatchHeightWidthChannel {
        batch: u32,
        height: u32,
        width: u32,
        channel: u32,
    },

    /// a.k.a. NC1HWC2
    BatchChannel1HeightWidthChannel2 {
        batch: u32,
        channel1: u32,
        height: u32,
        width: u32,
        channel2: u32,
    },

    Undefined(Box<[u32]>),
}

impl TensorDimension {
    pub fn multiplication(&self) -> u32 {
        match self {
            TensorDimension::BatchChannelHeightWidth {
                batch,
                channel,
                height,
                width,
            } => batch * channel * height * width,
            TensorDimension::BatchHeightWidthChannel {
                batch,
                height,
                width,
                channel,
            } => batch * channel * height * width,
            TensorDimension::BatchChannel1HeightWidthChannel2 {
                batch,
                channel1,
                height,
                width,
                channel2,
            } => batch * channel1 * height * width * channel2,
            TensorDimension::Undefined(dims) => {
                dims.iter().copied().reduce(|acc, e| acc * e).unwrap()
            }
        }
    }

    pub fn n_dims(&self) -> usize {
        match self {
            TensorDimension::BatchChannelHeightWidth { .. } => 4,
            TensorDimension::BatchHeightWidthChannel { .. } => 4,
            TensorDimension::BatchChannel1HeightWidthChannel2 { .. } => 5,
            TensorDimension::Undefined(dims) => dims.len(),
        }
    }

    pub fn get_fmt_type(&self) -> u32 {
        match self {
            TensorDimension::BatchChannelHeightWidth { .. } => _rknn_tensor_format_RKNN_TENSOR_NCHW,
            TensorDimension::BatchHeightWidthChannel { .. } => _rknn_tensor_format_RKNN_TENSOR_NHWC,
            TensorDimension::BatchChannel1HeightWidthChannel2 { .. } => {
                _rknn_tensor_format_RKNN_TENSOR_NC1HWC2
            }
            TensorDimension::Undefined(_) => _rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
        }
    }
}

impl Default for TensorDimension {
    fn default() -> Self {
        TensorDimension::BatchChannelHeightWidth {
            batch: 0,
            channel: 0,
            height: 0,
            width: 0,
        }
    }
}

#[derive(Debug, TryFromPrimitive, Clone, Copy, Default, PartialEq)]
#[repr(u32)]
pub enum TensorType {
    #[default]
    Float32 = _rknn_tensor_type_RKNN_TENSOR_FLOAT32,
    Float16 = _rknn_tensor_type_RKNN_TENSOR_FLOAT16,
    Int8 = _rknn_tensor_type_RKNN_TENSOR_INT8,
    UInt8 = _rknn_tensor_type_RKNN_TENSOR_UINT8,
    Int16 = _rknn_tensor_type_RKNN_TENSOR_INT16,
    UInt16 = _rknn_tensor_type_RKNN_TENSOR_UINT16,
    Int32 = _rknn_tensor_type_RKNN_TENSOR_INT32,
    Int64 = _rknn_tensor_type_RKNN_TENSOR_INT64,
    Bool = _rknn_tensor_type_RKNN_TENSOR_BOOL,
}

impl TensorType {
    pub fn byte_size(&self) -> usize {
        match self {
            TensorType::Float32 => size_of::<f32>(),
            TensorType::Float16 => size_of::<f16>(),
            TensorType::Int8 => size_of::<i8>(),
            TensorType::UInt8 => size_of::<u8>(),
            TensorType::Int16 => size_of::<i16>(),
            TensorType::UInt16 => size_of::<u16>(),
            TensorType::Int32 => size_of::<i32>(),
            TensorType::Int64 => size_of::<i64>(),
            TensorType::Bool => size_of::<bool>(),
        }
    }
}

// #[derive(Debug, TryFromPrimitive)]
// #[repr(u32)]
// pub enum QuantifyType {
//     None = _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE,
//     DynamicFloatingPoint = _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP,
//     AffineAsymmetric = _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC,
// }

#[derive(Debug, Default, Clone)]
pub enum QuantifyType {
    #[default]
    None,
    DynamicFixedPoint {
        fl: i8,
    },
    AffineAsymmetric {
        scale: f32,
    },
}

#[derive(Debug, Default, Clone)]
pub struct TensorAttr {
    index: u32,
    // n_dims: u32,
    // dims: [u32;16],
    // dims: Box<[u32]>,
    pub dims: TensorDimension,
    pub name: String,
    pub n_elems: u32,
    pub size: u32,
    // fmt: TensorFormat,
    pub data_type: TensorType,
    pub qnt_type: QuantifyType,
    pub w_stride: u32,
    pub size_with_stride: u32,
    pub pass_through: bool,
    pub h_stride: u32,
}

#[derive(Debug, Default, Clone)]
pub struct InputTensor(pub TensorAttr);

#[derive(Debug, Default, Clone)]
pub struct OutputTensor(TensorAttr);

// pub struct RkNNInput {
//
// }

/// Memory allocated by RkNN runtime,
/// used in zero-copy programming.
///
/// Allocate: `rknn_create_mem`
///
/// Destroy: `rknn_destroy_mem`
///
/// Output buffer requires `flush_cache`
pub struct RuntimeMemory {}

/// Memory **NOT** allocated by RkNN runtime,
/// used in zero-copy programming.
///
/// For generic consideration, this struct is designed to work with `Alloc` trait.
///
/// Allocate: `rknn_create_mem_from_fd` or `rknn_create_mem_from_phys`
///
/// Destroy: `rknn_destroy_mem`
///
/// Output buffer requires `flush_cache`
pub struct ExternalMemory {}
