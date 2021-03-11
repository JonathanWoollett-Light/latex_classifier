#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use latex_classifier::{construct::construct, models::*, segment::*, util::time};
    use simplers_optimization::Optimizer;

    use std::{fs, path::Path, time::Instant, usize};

    use image::{ImageBuffer, Luma};

    const DIF_SCALE: f64 = 0.2;

    // Best binarization parameters
    const EXTREME_LUMA_BOUNDARY: u8 = 8;
    const GLOBAL_LUMA_BOUNDARY: u8 = 179;

    const LOCAL_LUMA_BOUNDARY: u8 = 18;
    const LOCAL_FIELD_REACH: i32 = 35;
    const LOCAL_FIELD_SIZE: usize = 17;

    #[test]
    fn base() {
        // Runs segmentation
        // -----------------
        let segment = segment(
            "tests/images/8.jpg",
            EXTREME_LUMA_BOUNDARY,
            GLOBAL_LUMA_BOUNDARY,
            LOCAL_LUMA_BOUNDARY,
            LOCAL_FIELD_REACH,
            LOCAL_FIELD_SIZE65
        );

        assert!(false);

        // let classes: [&str; 15] = [
        //     "3", "2", "x", "7", "1", "2", "\\cdot ", "b", "+", "-", "y", "-", "-", "-", "\\cdot ",
        // ];
        // let bounds: Vec<Bound<usize>> = segment.into_iter().map(|(_, b)| b).collect();
        // let latex = construct(&classes, &bounds);
        // println!("latex :{}", latex);
    }

    // #[test]
    fn optimisation() {
        let start = Instant::now();
        let bounds: Vec<(f64, f64)> =
            vec![(0., 255.), (0., 255.), (0., 255.), (0., 50.), (1., 100.)];
        let iterations = 4000;
        // print!("COUNTER: ");
        let (min, v) = Optimizer::minimize(&outer_eval, &bounds, iterations);

        println!("{}", time(start));
        println!("{} : {:.2?}", min, v);

        for s in SAMPLES.iter() {
            save(
                s,
                v[0] as u8,
                v[1] as u8,
                v[2] as u8,
                v[3] as i32,
                v[4] as usize,
            );
        }
        // save("tests/images/3.jpg",v[0] as u8,v[1] as u8,v[2] as u8,v[3] as i32, v[4] as usize);

        assert!(false);

        static mut COUNTER: u32 = 0;
        static SAMPLES: [&str; 11] = [
            "tests/images/1.jpg",
            "tests/images/2.jpg",
            "tests/images/3.jpg",
            "tests/images/4.jpg",
            "tests/images/5.jpg",
            "tests/images/6.jpg",
            "tests/images/7.jpg",
            "tests/images/8.jpg",
            "tests/images/9.jpg",
            "tests/images/10.jpg",
            "tests/images/11.jpg",
        ];
        fn outer_eval(v: &[f64]) -> f64 {
            eval(
                &SAMPLES,
                v[0] as u8,
                v[1] as u8,
                v[2] as u8,
                v[3] as i32,
                v[4] as usize,
            )
        }
        fn eval(
            samples: &[&str],

            extreme_luma_boundary: u8,
            global_luma_boundary: u8,
            local_luma_boundary: u8,
            local_field_reach: i32,
            local_field_size: usize,
        ) -> f64 {
            samples
                .into_iter()
                .map(|path| {
                    let img = image::open(path).unwrap().to_luma8();

                    let dims = img.dimensions();
                    let size = (dims.0 as usize, dims.1 as usize);

                    let mut img_raw: Vec<u8> = img.into_raw();

                    let mut pixels = binarize(
                        size,
                        &mut img_raw,
                        extreme_luma_boundary,
                        global_luma_boundary,
                        local_luma_boundary,
                        local_field_reach,
                        local_field_size,
                    );

                    // unsafe {
                    //     let name = Path::new(path).file_stem().unwrap().to_str().unwrap();
                    //     print!("{} ",COUNTER);
                    //     if !Path::new("binary_imgs").exists() {
                    //         fs::create_dir("binary_imgs").unwrap();
                    //     }
                    //     let binary_image =
                    //         ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(dims.0, dims.1, img_raw.clone()).unwrap();
                    //     binary_image
                    //         .save(format!("binary_imgs/{}_{}.png", COUNTER, name))
                    //         .unwrap();
                    //     COUNTER += 1;
                    // }

                    // let grp_vec = get_pixel_groups(size, &mut pixels);
                    //println!("eval val: {}({},{})",(grp_vec.len() as i32 - *num_symbols as i32).abs() as u32,grp_vec.len(),*num_symbols);
                    // (grp_vec.len() as i32 - *num_symbols as i32).abs() as f64
                    // ((grp_vec.len() as i32 - *num_symbols as i32).abs() as f64).powf(DIF_SCALE)

                    let name = Path::new(path).file_stem().unwrap().to_str().unwrap();
                    img_mse(&img_raw, &format!("tests/images/targets/{}.jpg", name))
                })
                .sum::<f64>()
                / samples.len() as f64
        }
        fn img_mse(p_raw: &[u8], t_path: &str) -> f64 {
            let t_img = image::open(t_path).unwrap().to_luma8();
            p_raw
                .into_iter()
                .zip(t_img.into_raw().into_iter())
                .map(|(p, t)| (*p as f64 - t as f64).powf(2.))
                .sum::<f64>()
                / p_raw.len() as f64
        }
    }
    fn save(
        path: &str,
        extreme_luma_boundary: u8,
        global_luma_boundary: u8,
        local_luma_boundary: u8,
        local_field_reach: i32,
        local_field_size: usize,
    ) {
        let img = image::open(path).unwrap().to_luma8();

        let dims = img.dimensions();
        let size = (dims.0 as usize, dims.1 as usize);

        let mut img_raw: Vec<u8> = img.into_raw();

        let mut pixels = binarize(
            size,
            &mut img_raw,
            extreme_luma_boundary,
            global_luma_boundary,
            local_luma_boundary,
            local_field_reach,
            local_field_size,
        );

        let name = Path::new(path).file_stem().unwrap().to_str().unwrap();
        let test_name = format!("{}_test", name);
        if !Path::new("binary_imgs").exists() {
            fs::create_dir("binary_imgs").unwrap();
        }
        let binary_image =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(dims.0, dims.1, img_raw.clone()).unwrap();
        binary_image
            .save(format!("binary_imgs/{}.png", test_name))
            .unwrap();

        // Gets lists of pixels belonging to each symbol
        let pixel_lists: Vec<Vec<(usize, usize)>> = get_pixel_groups(size, &mut pixels);
        println!("groups: {}", pixel_lists.len());

        // Gets bounds, square bounds and square bounds scaling property for each symbol
        let bounds: Vec<(Bound<usize>, (Bound<i32>, i32))> = get_bounds(&pixel_lists);

        // Outputs borders image
        output_bounds(20, [255, 0, 0], path, &test_name, &img_raw, &bounds, size);
    }
}
