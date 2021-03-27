#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use latex_classifier::{models::*, segment::*, util::time};
    use simplers_optimization::Optimizer;

    use std::{fs, path::Path, time::Instant, usize};

    use image::{ImageBuffer, Luma, Rgb};
    use ndarray::Array;
    const DIF_SCALE: f64 = 0.2;

    // Best binarization parameters
    const EXTREME_BOUNDARY: u8 = 8;
    const GLOBAL_BOUNDARY: u8 = 179;
    const LOCAL_BOUNDARY: u8 = 18;
    const FIELD_REACH: usize = 35;
    const FIELD_SIZE: usize = 17;

    #[test]
    fn base() {
        // Runs segmentation
        // -----------------
        let string = String::from("tests/images/2.jpg");
        println!("string:\t{}", string);
        let vec = string.clone().into_bytes();
        println!("vec:\t{:.?}", vec);
        let c_str = Box::new(CArray::new(vec));

        let bin_params = Box::new(BinarizationParameters {
            extreme_boundary: EXTREME_BOUNDARY,
            global_boundary: GLOBAL_BOUNDARY,
            local_boundary: LOCAL_BOUNDARY,
            field_reach: FIELD_REACH,
            field_size: FIELD_SIZE,
        });
        let start = Instant::now();
        let rtn: *mut CReturn = segment_file(Box::into_raw(c_str), Box::into_raw(bin_params));
        println!("time: {}",time(start));
        //println!("time: {}",start.elapsed().as)

        // Output debug image
        unsafe {
            let pixels: &CArray<u8> = &(*rtn).pixels;

            let img = image::open(Path::new(&string))
                .expect("Opening image failed")
                .to_rgb8();
            let dims = img.dimensions();

            let pixel_vec: &[u8] = unsafe { std::slice::from_raw_parts((pixels).ptr, (pixels).size) };
            let image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(dims.0,dims.1, pixel_vec.to_vec()).expect("Image creation failed");
            image.save("test_image.png").expect("Image saving failed");
        }
        
        // assert!(false);

        // Iterate over symbols
        unsafe {
            let segment: &CArray<CArray<SymbolPixels>> = &(*rtn).symbols;
            println!("segment: {:.?}", (segment));
            let lines = std::slice::from_raw_parts((segment).ptr, (segment).size);
            println!("\t{:.?}", lines);
            for l in lines.iter() {
                let line = std::slice::from_raw_parts(l.ptr, l.size);
                for s in line.iter() {
                    println!("\t\t{:.?}", s);
                    let line = std::slice::from_raw_parts(s.pixels.ptr, s.pixels.size);
                    //let ndarray_arr = Array::from_shape_vec((24,24),line.to_vec()).expect("ndarray error");
                    for y in 0..24 {
                        for x in 0..24 {
                            if line[y*24+x] == 1 { print!("-"); } else { print!("*"); }
                        }
                        println!();
                    }
                    //println!("{:#.}",ndarray_arr);
                }
            }
        }

        assert!(false);

        // let classes: [&str; 15] = [
        //     "3", "2", "x", "7", "1", "2", "\\cdot ", "b", "+", "-", "y", "-", "-", "-", "\\cdot ",
        // ];
        // let bounds: Vec<Bound<usize>> = segment.into_iter().map(|(_, b)| b).collect();
        // let latex = construct(&classes, &bounds);
        // println!("latex :{}", latex);
    }

    // // #[test]
    // fn optimisation() {
    //     let start = Instant::now();
    //     let bounds: Vec<(f64, f64)> =
    //         vec![(0., 255.), (0., 255.), (0., 255.), (0., 50.), (1., 100.)];
    //     let iterations = 4000;
    //     // print!("COUNTER: ");
    //     let (min, v) = Optimizer::minimize(&outer_eval, &bounds, iterations);

    //     println!("{}", time(start));
    //     println!("{} : {:.2?}", min, v);

    //     for s in SAMPLES.iter() {
    //         save(
    //             s,
    //             v[0] as u8,
    //             v[1] as u8,
    //             v[2] as u8,
    //             v[3] as usize,
    //             v[4] as usize,
    //         );
    //     }
    //     // save("tests/images/3.jpg",v[0] as u8,v[1] as u8,v[2] as u8,v[3] as i32, v[4] as usize);

    //     assert!(false);

    //     static mut COUNTER: u32 = 0;
    //     static SAMPLES: [&str; 11] = [
    //         "tests/images/1.jpg",
    //         "tests/images/2.jpg",
    //         "tests/images/3.jpg",
    //         "tests/images/4.jpg",
    //         "tests/images/5.jpg",
    //         "tests/images/6.jpg",
    //         "tests/images/7.jpg",
    //         "tests/images/8.jpg",
    //         "tests/images/9.jpg",
    //         "tests/images/10.jpg",
    //         "tests/images/11.jpg",
    //     ];
    //     fn outer_eval(v: &[f64]) -> f64 {
    //         eval(
    //             &SAMPLES,
    //             v[0] as u8,
    //             v[1] as u8,
    //             v[2] as u8,
    //             v[3] as usize,
    //             v[4] as usize,
    //         )
    //     }
    //     fn eval(
    //         samples: &[&str],

    //         extreme_boundary: u8,
    //         global_boundary: u8,
    //         local_boundary: u8,
    //         field_reach: usize,
    //         field_size: usize,
    //     ) -> f64 {
    //         samples
    //             .into_iter()
    //             .map(|path| {
    //                 let img = image::open(path).unwrap().to_luma8();

    //                 let dims = img.dimensions();
    //                 let size = (dims.0 as usize, dims.1 as usize);

    //                 let mut img_raw: Vec<u8> = img.into_raw();

    //                 let _pixels = binarize(
    //                     size,
    //                     &mut img_raw,
    //                     extreme_boundary,
    //                     global_boundary,
    //                     local_boundary,
    //                     field_reach,
    //                     field_size,
    //                 );

    //                 // unsafe {
    //                 //     let name = Path::new(path).file_stem().unwrap().to_str().unwrap();
    //                 //     print!("{} ",COUNTER);
    //                 //     if !Path::new("binary_imgs").exists() {
    //                 //         fs::create_dir("binary_imgs").unwrap();
    //                 //     }
    //                 //     let binary_image =
    //                 //         ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(dims.0, dims.1, img_raw.clone()).unwrap();
    //                 //     binary_image
    //                 //         .save(format!("binary_imgs/{}_{}.png", COUNTER, name))
    //                 //         .unwrap();
    //                 //     COUNTER += 1;
    //                 // }

    //                 // let grp_vec = get_pixel_groups(size, &mut pixels);
    //                 //println!("eval val: {}({},{})",(grp_vec.len() as i32 - *num_symbols as i32).abs() as u32,grp_vec.len(),*num_symbols);
    //                 // (grp_vec.len() as i32 - *num_symbols as i32).abs() as f64
    //                 // ((grp_vec.len() as i32 - *num_symbols as i32).abs() as f64).powf(DIF_SCALE)

    //                 let name = Path::new(path).file_stem().unwrap().to_str().unwrap();
    //                 img_mse(&img_raw, &format!("tests/images/targets/{}.jpg", name))
    //             })
    //             .sum::<f64>()
    //             / samples.len() as f64
    //     }
    //     fn img_mse(p_raw: &[u8], t_path: &str) -> f64 {
    //         let t_img = image::open(t_path).unwrap().to_luma8();
    //         p_raw
    //             .into_iter()
    //             .zip(t_img.into_raw().into_iter())
    //             .map(|(p, t)| (*p as f64 - t as f64).powf(2.))
    //             .sum::<f64>()
    //             / p_raw.len() as f64
    //     }
    // }
    // fn save(
    //     path: &str,
    //     extreme_luma_boundary: u8,
    //     global_luma_boundary: u8,
    //     local_luma_boundary: u8,
    //     local_field_reach: usize,
    //     local_field_size: usize,
    // ) {
    //     let img = image::open(path).unwrap().to_luma8();

    //     let dims = img.dimensions();
    //     let size = (dims.0 as usize, dims.1 as usize);

    //     let mut img_raw: Vec<u8> = img.into_raw();

    //     let pixels = binarize(
    //         size,
    //         &mut img_raw,
    //         extreme_luma_boundary,
    //         global_luma_boundary,
    //         local_luma_boundary,
    //         local_field_reach,
    //         local_field_size,
    //     );

    //     let name = Path::new(path).file_stem().unwrap().to_str().unwrap();
    //     let test_name = format!("{}_test", name);
    //     if !Path::new("binary_imgs").exists() {
    //         fs::create_dir("binary_imgs").unwrap();
    //     }
    //     let binary_image =
    //         ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(dims.0, dims.1, img_raw.clone()).unwrap();
    //     binary_image
    //         .save(format!("binary_imgs/{}.png", test_name))
    //         .unwrap();

    //     // Gets lists of pixels belonging to each symbol
    //     let pixel_lists: Vec<Vec<(usize, usize)>> = get_pixel_groups(size, pixels);
    //     println!("groups: {}", pixel_lists.len());

    //     // Gets bounds, square bounds and square bounds scaling property for each symbol
    //     let bounds: Vec<(Bound<usize>, (Bound<i32>, i32))> = get_bounds(&pixel_lists);

    //     // Outputs borders image
    //     output_bounds(20, [255, 0, 0], path, &test_name, &img_raw, &bounds, size);
    // }
}
