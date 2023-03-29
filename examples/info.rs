use rknn2;
use rknn2::RkNN;
use pretty_env_logger;
use log::{info};

fn main() {
    pretty_env_logger::init();
    // let instance = RkNN::new_from_path("/home/shiroki/hand_landmark_full.rknn").unwrap();
    let instance = RkNN::new_from_path("/home/shiroki/palm_detection_full.rknn").unwrap();
    let (api_ver, drv_ver) = instance.query_sdk_version().unwrap();
    info!("Version: API {}, Driver {}", api_ver, drv_ver);

    // dbg!(instance.query_sdk_version().unwrap());
    
    // dbg!(instance.query_in_out_num()).unwrap();
    // dbg!(instance.query_inputs_outputs()).unwrap();
    let (inputs, outputs) = instance.query_inputs_outputs().unwrap();
    info!("Inputs: {:?}", inputs);
    info!("Outputs: {:#?}", outputs);
}
