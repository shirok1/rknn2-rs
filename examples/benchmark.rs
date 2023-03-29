use half::f16;
use image::Rgb;
use imageproc::rect::Rect;
use log::info;
use pretty_env_logger;
use rknn2;
use rknn2::RkNN;

fn main() {
    pretty_env_logger::init_timed();

    let dynamic_image = image::open("/home/shiroki/rknn2-rs/examples/test_hand.bmp").unwrap();
    let img = dynamic_image.as_rgb8().unwrap();
    // img.as_bytes()

    // let instance = RkNN::new_from_path("/home/shiroki/hand_landmark_full.rknn").unwrap();
    let instance = RkNN::new_from_path("/home/shiroki/palm_detection_full.rknn").unwrap();
    let (api_ver, drv_ver) = instance.query_sdk_version().unwrap();
    info!("Version: API {}, Driver {}", api_ver, drv_ver);

    // dbg!(instance.query_sdk_version().unwrap());

    // dbg!(instance.query_in_out_num()).unwrap();
    // dbg!(instance.query_inputs_outputs()).unwrap();
    let (inputs, outputs) = instance.query_inputs_outputs(false).unwrap();
    info!("Inputs: {:?}", inputs);
    info!("Outputs: {:#?}", outputs);

    let input = &inputs[0];
    // let buf = Vec::with_capacity(input.0.size_with_stride as usize);
    let mut buf = Vec::new();
    buf.resize(input.0.size_with_stride as usize, f16::from_f32(0.0));
    // for value in img.as_rgb8().unwrap().iter() {
    //
    // }
    for (x, y, Rgb([r, g, b])) in img.enumerate_pixels() {
        buf[(x + 192 * y) as usize] = f16::from(*r) / f16::from_f32_const(256f32);
        buf[(x + 192 * y + 192 * 192) as usize] = f16::from(*g) / f16::from_f32_const(256f32);
        buf[(x + 192 * y + 192 * 192 * 2) as usize] = f16::from(*b) / f16::from_f32_const(256f32);
    }
    // let buf = img.iter().map(|value|
    //     f16::from(*value) / f16::from_f32_const(256f32)).collect::<Vec<_>>();
    // let buf = [0u8;input.0.size_with_stride as usize];

    instance
        .inputs_set(&[(input, bytemuck::cast_slice(buf.as_slice()))])
        .unwrap();

    // for _ in 0..120 {
    //     instance.run().unwrap();
    // }
    instance.run().unwrap();

    let anchors = generate_anchors();
    // info!("Anchors' len = {}", anchors.len());

    let out = instance
        .outputs_get_allocated_by_runtime(outputs.as_slice())
        .unwrap();
    // info!("Out: {:#?}", out);
    let outs = out.iter().collect::<Vec<_>>();
    let boxes: &[f32] = bytemuck::cast_slice(outs[0]);
    let scores: &[f32] = bytemuck::cast_slice(outs[1]);
    let mut drawing = img.clone();
    // for (anchor_x, anchor_y) in anchors.iter() {
    //     imageproc::drawing::draw_filled_circle_mut(
    //         &mut drawing,
    //         ((*anchor_x*192.0) as i32,
    //          (*anchor_y*192.0) as i32), 2, Rgb([0, 255, 0]));
    //     // drawing = imageproc::drawing::draw_hollow_rect(&drawing, Rect::at(
    //     //     *anchor_x as i32,
    //     //     *anchor_y as i32).of_size(
    //     //     w as u32,
    //     //     h as u32,
    //     // ), Rgb([255,0,0]));
    // }
    for (i, ((box_, score), (anchor_x, anchor_y))) in
        boxes.chunks(18).zip(scores).zip(anchors).enumerate()
    {
        fn sigmoid(v: f32) -> f32 {
            if v < -40.0 {
                0.0
            } else if v > 40.0 {
                1.0
            } else {
                1.0 / (1.0 + f32::exp(-v))
            }
        }
        let sigmoid_score = sigmoid(*score);

        if sigmoid_score > 0.55f32 {
            let cx = box_[0] / 192.0 + anchor_x;
            let cy = box_[1] / 192.0 + anchor_y;
            let w = box_[2] / 192.0;
            let h = box_[3] / 192.0;
            imageproc::drawing::draw_filled_circle_mut(
                &mut drawing,
                ((anchor_x * 192.0) as i32, (anchor_y * 192.0) as i32),
                1,
                Rgb([0, 255, 0]),
            );

            let top_left = (cx - w * 0.5, cy - h * 0.5);
            let btm_right = (cx + w * 0.5, cy + h * 0.5);
            info!("Box {}: (cx, cy) {:?}, (w, h) {:?}", i, (cx, cy), (w, h));
            info!(
                "Box {}: score {} tl {:?}, br {:?}",
                i, sigmoid_score, top_left, btm_right
            );
            if w > 0.1 && h > 0.1 {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut drawing,
                    Rect::at(-top_left.1 as i32, -top_left.0 as i32).of_size(w as u32, h as u32),
                    Rgb([255, 0, 0]),
                );
            }
            for j in 0..7 {
                let lx = box_[4 + 2 * j + 0] / 192.0 + anchor_x;
                let ly = box_[4 + 2 * j + 1] / 192.0 + anchor_y;
                info!("Landmark {}: {:?}", j, (lx, ly));
            }
        }
    }
    drawing
        .save("/home/shiroki/rknn2-rs/examples/test_hand_out.bmp")
        .unwrap();
}

fn calculate_scale(min_scale: f32, max_scale: f32, stride_index: i32, num_strides: i32) -> f32 {
    if num_strides == 1 {
        (min_scale + max_scale) * 0.5
    } else {
        min_scale + (max_scale - min_scale) * 1.0 * stride_index as f32 / (num_strides - 1) as f32
    }
}

fn generate_anchors() -> Vec<(f32, f32)> {
    let mut anchors: Vec<(f32, f32)> = Vec::new();
    let mut layer_id = 0;
    const STRIDES: [i32; 4] = [8, 16, 16, 16];
    const MIN_SCALE: f32 = 0.1484375f32;
    const MAX_SCALE: f32 = 0.75f32;

    let input_size_height = 192i32;
    let input_size_width = 192i32;
    let anchor_offset_x = 0.5;
    let anchor_offset_y = 0.5;

    while layer_id < STRIDES.len() {
        let mut anchor_height: Vec<f32> = Vec::new();
        let mut anchor_width: Vec<f32> = Vec::new();
        let mut aspect_ratios: Vec<f32> = Vec::new();
        let mut scales: Vec<f32> = Vec::new();

        let mut last_same_stride_layer = layer_id;
        while last_same_stride_layer < STRIDES.len()
            && STRIDES[last_same_stride_layer] == STRIDES[layer_id]
        {
            let scale = calculate_scale(
                MIN_SCALE,
                MAX_SCALE,
                last_same_stride_layer as i32,
                STRIDES.len() as i32,
            );
            {
                // for aspect_ratio_id in 0..anchor_params.aspect_ratios.len() {
                //     aspect_ratios.push(anchor_params.aspect_ratios[aspect_ratio_id]);
                //     scales.push(scale);
                // }
                aspect_ratios.push(1f32);
                scales.push(scale);

                let scale_next = if last_same_stride_layer == STRIDES.len() - 1 {
                    1.0
                } else {
                    calculate_scale(
                        MIN_SCALE,
                        MAX_SCALE,
                        last_same_stride_layer as i32 + 1,
                        STRIDES.len() as i32,
                    )
                };
                scales.push((scale * scale_next).sqrt());
                aspect_ratios.push(1.0);
            }
            last_same_stride_layer += 1;
        }

        for i in 0..aspect_ratios.len() {
            let ratio_sqrts = aspect_ratios[i].sqrt();
            anchor_height.push(scales[i] / ratio_sqrts);
            anchor_width.push(scales[i] * ratio_sqrts);
        }

        let feature_map_height: i32 =
            (input_size_height as f32 / STRIDES[layer_id] as f32).ceil() as i32;
        let feature_map_width: i32 =
            (input_size_width as f32 / STRIDES[layer_id] as f32).ceil() as i32;

        for y in 0..feature_map_height {
            for x in 0..feature_map_width {
                for _anchor_id in 0..anchor_height.len() {
                    let x_center = (x as f32 + anchor_offset_x) / feature_map_width as f32;
                    let y_center = (y as f32 + anchor_offset_y) / feature_map_height as f32;

                    // let mut new_anchor = Anchor::default();
                    // new_anchor.x_center = x_center;
                    // new_anchor.y_center = y_center;
                    //
                    // new_anchor.w = 1.0;
                    // new_anchor.h = 1.0;
                    //
                    // anchors.push(new_anchor);
                    anchors.push((x_center, y_center));
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    anchors
}
