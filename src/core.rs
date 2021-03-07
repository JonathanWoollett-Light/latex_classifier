use crate::models::*;

use std::{cmp, collections::VecDeque, fs, path::Path, time::Instant, usize};

use image::{imageops::FilterType, ImageBuffer, Luma};
use num_format::{Locale, ToFormattedString};

const SCALE: u32 = 20u32;
const ROW_CLEARANCE: f32 = 0.3f32;

const GLOBAL_LUMA_BOUNDARY: u8 = 30;
const LOCAL_LUMA_BOUNDARY: u8 = 50;
const LOCAL_FIELD_SIZE: i32 = 40;

// Returns symbol images and bounds
pub fn segment(path: &str) -> Vec<(Vec<u8>, Bound<usize>)> {
    #[derive(Clone, Eq, PartialEq)]
    enum Pixel {
        White,
        Black,
        Assigned,
    }

    #[cfg(debug_assertions)]
    let start = Instant::now();

    // Open the image to segment (in this case it will reside within the `test_imgs` directory)
    let img = image::open(path).unwrap().to_luma8();

    // Gets dimensions of the image
    let dims = img.dimensions();
    let size = (dims.0 as usize, dims.1 as usize);

    // Gets raw pixel values from the image.
    let mut img_raw: Vec<u8> = img.into_raw();

    // 2d vector of size of image, where each pixel will be labelled white/black (and later used in forest fire)
    let mut pixels: Vec<Vec<Pixel>> = raw_to_binary_2d_vector(size, &mut img_raw);

    // Gets name of image file ('some_image.jpg' -> 'some_image')
    let name = Path::new(path).file_stem().unwrap().to_str().unwrap();

    // Outputs binary image
    // #[cfg(debug_assertions)] // OFF FOR TESTING
    {
        if !Path::new("binary_imgs").exists() {
            fs::create_dir("binary_imgs").unwrap();
        }
        let binary_image =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(dims.0, dims.1, img_raw.clone()).unwrap();
        binary_image
            .save(format!("binary_imgs/{}.png", name))
            .unwrap();
    }

    // Gets lists of pixels belonging to each symbol
    let pixel_lists: Vec<Vec<(usize, usize)>> = get_pixels_in_symbols(size, &mut pixels);

    // Gets bounds, square bounds and square bounds scaling property for each symbol
    let bounds: Vec<(Bound<usize>, (Bound<i32>, i32))> = get_bounds(&pixel_lists);

    // Outputs borders image
    // #[cfg(debug_assertions)] // TEMP OFF FOR TESTING
    output_bounds(2, [255, 0, 0], path, name, &img_raw, &bounds, size);

    // Gets scaled pixels belonging to each symbol
    let symbols: Vec<Vec<u8>> = get_symbols(&pixel_lists, &bounds);

    // Outputs symbol images
    // #[cfg(debug_assertions)] // TEMP OFF FOR TESTING
    output_symbols(&symbols, name);

    #[cfg(debug_assertions)]
    println!("{} : Finished segmentation", time(start));

    return symbols
        .into_iter()
        .zip(bounds.into_iter())
        .map(|(s, b)| (s, b.0))
        .collect();

    fn raw_to_binary_2d_vector(
        (width, height): (usize, usize),
        img_raw: &mut Vec<u8>,
    ) -> Vec<Vec<Pixel>> {
        #[cfg(debug_assertions)]
        let start = Instant::now();

        // Intialises 2d vector of size height*width with Pixel::White.
        let mut pixels: Vec<Vec<Pixel>> = vec![vec!(Pixel::White; width as usize); height as usize];

        #[cfg(debug_assertions)]
        println!(
            "{} * {} = {}",
            width.to_formatted_string(&Locale::en),
            height.to_formatted_string(&Locale::en),
            img_raw.len().to_formatted_string(&Locale::en)
        );

        // Gets average luma among pixels
        //  Uses `.fold()` instead of `.sum()` as sum of values will likely exceed `u8:MAX`
        // let global_luma: u8 =
        //     (img_raw.iter().fold(0u32, |sum, &x| sum + x as u32) / img_raw.len() as u32) as u8;

        #[cfg(debug_assertions)]
        let mut skipped: u32 = 0;

        // O(n*(2p)^2) // n = img_raw.len(), p = LOCAL_FIELD_SIZE
        let mut local_luma: Vec<u8> = vec![u8::default();img_raw.len()];
        for y in 0..height as i32 {

            //print!(" y:{}",y); // TO REMOVE

            let y_r = cmp::max(y - LOCAL_FIELD_SIZE, 0)
                ..cmp::min(y + LOCAL_FIELD_SIZE + 1, height as i32);
            
            let y_size = y_r.end - y_r.start;
            for x in 0..width as i32 {
                //print!(" x:{}",x); // TO REMOVE

                let i = (y * width as i32 + x) as usize;

                // If pixel is extremely dark or extremely light, we need not bother with the
                //  nuance of calculating the local average luminance as binarization should be 
                //  obvious.
                if img_raw[i] < GLOBAL_LUMA_BOUNDARY || img_raw[i] > 255-GLOBAL_LUMA_BOUNDARY {
                    #[cfg(debug_assertions)]
                    {
                        skipped += 1;
                    }
                    continue;
                }

                //print!(" i:{}",i);


                let x_r = cmp::max(x - LOCAL_FIELD_SIZE, 0)
                    ..cmp::min(x + LOCAL_FIELD_SIZE + 1, width as i32);
                let x_size = x_r.end - x_r.start;

                //print!(" x_r:{:.?}",x_r);

                #[rustfmt::skip]
                let total: u32 = y_r.clone().map(|yl| {
                    x_r.clone().map(|xl| {
                        let il = yl * width as i32 + xl;
                        img_raw[il as usize] as u32
                    }).sum::<u32>()
                }).sum();

                // let total: u32 = y_r.clone().map(|yl| {
                //     let start = (yl * width as i32 + x) as usize;
                //     let end = start + x_size as usize;
                //     img_raw[start..end].iter().map(|x|*x as u32).sum::<u32>()
                // }).sum();

                //print!(" total:{}",total);

                //println!("{},{} \n {:.?},{:.?} \n {},{}",y_size,x_size,y_r,x_r,y,x);

                // if (y_size * x_size) == 0 || total / (y_size * x_size) as u16 > u8::max_value() as u16 {
                //     println!("{},{} \n {:.?},{:.?} \n {},{}",y_size,x_size,y_r,x_r,y,x);
                // }
                local_luma[i] = (total / (y_size * x_size) as u32) as u8;
                
                //println!(" local_luma[i]:{}",local_luma[i]);
            }
        }

        #[cfg(debug_assertions)]
        println!("{} : Calculated local luminance fields ({})", time(start),skipped.to_formatted_string(&Locale::en));

        #[cfg(debug_assertions)]
        let start = Instant::now();

        for (y, row) in pixels.iter_mut().enumerate() {
            for (x, p) in row.iter_mut().enumerate() {
                let i = y * width + x; // Index of pixel of coordinates (x,y) in `img_raw`
                
                if img_raw[i] < GLOBAL_LUMA_BOUNDARY  { 
                    img_raw[i] = 0;
                    *p = Pixel::Black;
                    continue;
                }
                if img_raw[i] > 255-GLOBAL_LUMA_BOUNDARY {
                    img_raw[i] = 255;
                    // pixels `Pixel::White` by default
                    continue;
                }
                if let Some(local) = local_luma[i].checked_sub(LOCAL_LUMA_BOUNDARY) {
                    if img_raw[i] < local {
                        img_raw[i] = 0;
                        *p = Pixel::Black;
                        continue;
                    }
                }
                if let Some(local) = local_luma[i].checked_add(LOCAL_LUMA_BOUNDARY) {
                    if img_raw[i] > local {
                        img_raw[i] = 255;
                        // pixels `Pixel::White` by default
                        continue;
                    }
                }
                // White is the negative (false/0) colour in our binarization, thus this is our else case
                img_raw[i] = 255;
                // pixels `Pixel::White` by default
            }
        }

        #[cfg(debug_assertions)]
        println!("{} : Converted image to binary", time(start));

        pixels
    }
    fn get_pixels_in_symbols(
        (width, height): (usize, usize),
        pixels: &mut Vec<Vec<Pixel>>,
    ) -> Vec<Vec<(usize, usize)>> {
        #[cfg(debug_assertions)]
        let start = Instant::now();

        // List of lists of pixels belonging to each symbol.
        let mut pixel_lists: Vec<Vec<(usize, usize)>> = Vec::new();

        // Iterates through pixels
        for y in 0..height {
            for x in 0..width {
                if pixels[y][x] == Pixel::Black {
                    // Pushes new list to hold pixels belonging to this newly found symbol
                    pixel_lists.push(Vec::new());

                    // Triggers the forest fire algorithm
                    let last_index = pixel_lists.len() - 1;
                    forest_fire(pixels, width, height, x, y, &mut pixel_lists[last_index]);
                }
            }
        }

        #[cfg(debug_assertions)]
        println!("{} : Fill finished", time(start));

        return pixel_lists;

        fn forest_fire(
            pixels: &mut Vec<Vec<Pixel>>,
            width: usize,
            height: usize,
            x: usize,
            y: usize,
            pixel_list: &mut Vec<(usize, usize)>,
        ) {
            // Push 1st pixel to symbol
            pixel_list.push((x, y));
            // Sets 1st pixels symbol number
            pixels[y][x] = Pixel::Assigned;
            // Initialises queue for forest fire
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            // Pushes 1st pixel to queue
            queue.push_back((x, y));

            // While there is value in queue (effectively while !queue.is_empty())
            while let Some(n) = queue.pop_front() {
                // +x
                if n.0 < width - 1 {
                    let (x, y) = (n.0 + 1, n.1);
                    pixels[y][x] = run_pixel(x, y, pixels, pixel_list, &mut queue);
                }
                // -x
                if n.0 > 0 {
                    let (x, y) = (n.0 - 1, n.1);
                    pixels[y][x] = run_pixel(x, y, pixels, pixel_list, &mut queue);
                }
                // +y
                if n.1 < height - 1 {
                    let (x, y) = (n.0, n.1 + 1);
                    pixels[y][x] = run_pixel(x, y, pixels, pixel_list, &mut queue);
                }
                // -y
                if n.1 > 0 {
                    let (x, y) = (n.0, n.1 - 1);
                    pixels[y][x] = run_pixel(x, y, pixels, pixel_list, &mut queue);
                }
            }
            fn run_pixel(
                x: usize,
                y: usize,
                pixels: &Vec<Vec<Pixel>>,
                pixel_list: &mut Vec<(usize, usize)>,
                queue: &mut VecDeque<(usize, usize)>,
            ) -> Pixel {
                if pixels[y][x] == Pixel::Black {
                    // If black pixel unassigned to symbol
                    pixel_list.push((x, y)); // Push pixel to symbol
                    queue.push_back((x, y)); // Enqueue pixel for forest fire algorithm
                    return Pixel::Assigned; // Return value to set `symbols[y][x]` to
                }
                return Pixel::White; // Return value to set `symbols[y][x]` to
            }
        }
    }

    // Returns tupel of: 2d vec of pixels in symbol, Bounds of symbol
    fn get_symbols(
        pixel_lists: &[Vec<(usize, usize)>],
        bounds: &[(Bound<usize>, (Bound<i32>, i32))],
    ) -> Vec<Vec<u8>> {
        #[cfg(debug_assertions)]
        let start = Instant::now();
        let mut symbols: Vec<Vec<u8>> = Vec::with_capacity(pixel_lists.len());

        for i in 0..pixel_lists.len() {
            // Sets `min_actuals` as minimum bounds and `bounds` as square bounds.
            let (bound, (sqr_bound, sqr_diff)) = &bounds[i];

            // Calculates height and width of image using square bounds
            let height: usize = (sqr_bound.max.y - sqr_bound.min.y + 1) as usize;
            let width: usize = (sqr_bound.max.x - sqr_bound.min.x + 1) as usize;

            // Constructs list to hold symbol image
            let mut symbol = vec![vec!(255u8; width); height];

            // Iterates over pixels belonging to symbol
            for &(x, y) in pixel_lists[i].iter() {
                // Sets x,y coordinates scaled to square bounds from original x,y coordinates.
                let (nx, ny) = if *sqr_diff < 0 {
                    ((x as i32 - sqr_diff) as usize, y)
                } else {
                    (x, (y as i32 + sqr_diff) as usize)
                };

                // Sets pixel in symbol image list
                symbol[ny - bound.min.y][nx - bound.min.x] = 0u8;
            }

            // Constructs image buffer from symbol vector
            let symbol_image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                width as u32,
                height as u32,
                symbol.into_iter().flatten().collect(),
            )
            .unwrap();

            // Scales image buffer
            let scaled_image =
                image::imageops::resize(&symbol_image, SCALE, SCALE, FilterType::Triangle);

            // Sets list to image buffer and carries out binarization.
            //  Running basic binarization here is necessary as scaling will have blurred some pixels.
            //  Basic binarization should also be proficient as the blurring will be minor.
            let binary_vec: Vec<u8> = scaled_image
                .into_raw()
                .into_iter()
                .map(|p| if p < 220 { 0 } else { 1 })
                .collect();

            // Pushes the scaled symbol list to the symbols list.
            symbols.push(binary_vec);
        }
        #[cfg(debug_assertions)]
        println!("{} : Symbols set", time(start));

        symbols
    }
    fn output_symbols(symbols: &[Vec<u8>], name: &str) {
        #[cfg(debug_assertions)]
        let start = Instant::now();
        // Create folder
        if !Path::new("split").exists() {
            fs::create_dir("split").unwrap();
        }
        // If folder exists, empty it.
        let path = format!("split/{}", name);
        if Path::new(&path).exists() {
            fs::remove_dir_all(&path).unwrap(); // Delete folder
        }
        fs::create_dir(&path).unwrap(); // Create folder

        for (i, symbol) in symbols.iter().enumerate() {
            let symbol_image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                SCALE,
                SCALE,
                symbol
                    .iter()
                    .map(|&x| if x == 1 { 255 } else { 0 })
                    .collect(),
            )
            .unwrap();

            let path = format!("split/{}/{}.png", name, i);

            symbol_image.save(path).unwrap();
        }

        //Export bounds
        #[cfg(debug_assertions)]
        println!("{} : Symbols output", time(start));
    }
    fn get_bounds(pixel_lists: &[Vec<(usize, usize)>]) -> Vec<(Bound<usize>, (Bound<i32>, i32))> {
        let start = Instant::now();
        // Gets bounds
        let mut bounds: Vec<Bound<usize>> = Vec::new();
        let mut sqr_bounds: Vec<(Bound<i32>, i32)> = Vec::new();

        // Iterate across all pixels in symbol
        for symbol in pixel_lists {
            let (mut lower_x, mut lower_y) = symbol[0];
            let (mut upper_x, mut upper_y) = symbol[0];
            for p in symbol.iter().skip(1) {
                let &(x, y) = p;

                if x < lower_x {
                    lower_x = x;
                } else if x > upper_x {
                    upper_x = x;
                }

                if y < lower_y {
                    lower_y = y;
                } else if y > upper_y {
                    upper_y = y;
                }
            }

            // Gets square bounds centred on original bounds
            bounds.push(Bound::from(((lower_x, lower_y), (upper_x, upper_y))));
            sqr_bounds.push(square_indxs(lower_x, lower_y, upper_x, upper_y));
        }

        #[cfg(debug_assertions)]
        println!("{} : Bounds set", time(start));

        return bounds.into_iter().zip(sqr_bounds.into_iter()).collect();

        // talk about adding this
        fn square_indxs(
            lower_x: usize,
            lower_y: usize,
            upper_x: usize,
            upper_y: usize,
        ) -> (Bound<i32>, i32) {
            let (view_width, view_height) = (upper_x - lower_x, upper_y - lower_y);

            let dif: i32 = view_width as i32 - view_height as i32;
            let dif_by_2 = dif / 2;
            // If width > height
            if dif > 0 {
                (
                    Bound::from((
                        (lower_x as i32, lower_y as i32 - dif_by_2),
                        (upper_x as i32, upper_y as i32 + dif_by_2),
                    )),
                    dif_by_2,
                )
            }
            // If width < height (if 0 has no affect)
            else {
                (
                    Bound::from((
                        (lower_x as i32 + dif_by_2, lower_y as i32),
                        (upper_x as i32 - dif_by_2, upper_y as i32),
                    )),
                    dif_by_2,
                )
            }
        }
    }
    fn output_bounds(
        spacing: usize,
        colour: [u8; 3],
        path: &str,
        name: &str,
        symbols: &[u8],
        bounds: &[(Bound<usize>, (Bound<i32>, i32))],
        (width, height): (usize, usize),
    ) {
        #[cfg(debug_assertions)]
        let start = Instant::now();

        let i32_sp = spacing as i32;
        let border_bounds: Vec<Bound<i32>> = bounds
            .iter()
            .map(|(_, (b, _))| {
                Bound::from((
                    (b.min.x - i32_sp, b.min.y - i32_sp),
                    (b.max.x + i32_sp, b.max.y + i32_sp),
                ))
            })
            .collect();

        // Opens another version of image
        let mut border_img = image::open(path).unwrap().into_rgb8();
        for (x, y, pixel) in border_img.enumerate_pixels_mut() {
            let val = symbols[((y as usize) * width) + (x as usize)];
            *pixel = image::Rgb([val, val, val]);
        }

        let (width, height) = (width as i32, height as i32);
        // Sets borders
        let border_pixel = image::Rgb(colour); // Pixel to use as border
        for b in border_bounds.iter() {
            // Sets horizontal borders
            let max_x_indx = if b.max.x < width { b.max.x } else { width };
            let min_x_indx = if b.min.x < 0 { 0 } else { b.min.x };
            //println!("{} / {}",max_x_indx,min_x_indx);
            for i in min_x_indx..max_x_indx {
                if b.min.y >= 0 {
                    *border_img.get_pixel_mut(i as u32, b.min.y as u32) = border_pixel;
                }
                if b.max.y < height {
                    *border_img.get_pixel_mut(i as u32, b.max.y as u32) = border_pixel;
                }
            }
            // Sets vertical borders
            let max_y_indx = if b.max.y < height { b.max.y } else { height };
            let min_y_indx = if b.min.y < 0 { 0 } else { b.min.y };
            //println!("{} / {}",max_y_indx,min_y_indx);
            for i in min_y_indx..max_y_indx {
                if b.min.x >= 0 {
                    *border_img.get_pixel_mut(b.min.x as u32, i as u32) = border_pixel;
                }
                if b.max.x < width {
                    *border_img.get_pixel_mut(b.max.x as u32, i as u32) = border_pixel;
                }
            }
            // Sets bottom corner border
            if b.max.x < width && b.max.y < height {
                *border_img.get_pixel_mut(b.max.x as u32, b.max.y as u32) = border_pixel;
            }
        }
        if !Path::new("borders").exists() {
            fs::create_dir("borders").unwrap();
        }
        border_img.save(format!("borders/{}.png", name)).unwrap();

        #[cfg(debug_assertions)]
        println!("{} : Bounds output", time(start));
    }
}

pub fn construct(classes: &[&str], bounds: &[Bound<usize>]) -> String {
    // Struct for symbol
    #[derive(Clone, Debug)]
    struct Symbol {
        class: String,
        bounds: Bound<usize>,
    }
    // Struct for row
    #[derive(Debug)]
    struct Row {
        center: usize,
        height: usize,
        sum: usize,
        symbols: Vec<Symbol>,
        superscript: Option<*mut Row>,
        subscript: Option<*mut Row>,
        parent: Option<*mut Row>,
    }
    impl Row {
        fn print_symbols(&self) -> String {
            format!(
                "[{}]",
                self.symbols
                    .iter()
                    .map(|s| format!("{} ", s.class))
                    .collect::<String>()
            )
        }
    }
    #[cfg(debug_assertions)]
    let start = Instant::now();

    // Converts given symbols and bounds into `Symbol` structs
    let mut combined: Vec<Symbol> = classes
        .iter()
        .zip(bounds.iter())
        .map(|(&class, bound)| Symbol {
            class: class.to_string(),
            bounds: bound.clone(),
        })
        .collect();

    // Sorts symbols by min x bound, ordering symbols horizontally
    // O(n log n)
    combined.sort_by_key(|a| ((a.bounds).min).x);

    // min x and y out of all symbols
    let min_x: usize = combined[0].bounds.min.x; // O(1)
    let min_y: usize = bounds
        .iter()
        .min_by_key(|b| b.min.y)
        .expect("Bounds empty")
        .min
        .y; // O(n)

    let origin = Point { x: min_x, y: min_y };

    // Subtract mins (`origin`) from bounds of all symbols
    for row in combined.iter_mut() {
        row.bounds -= origin;
    }

    // Calculates center y coordinate of each symbol
    let y_centers: Vec<usize> = combined.iter().map(|s| s.bounds.y_center()).collect();

    // Initializes rows, 1st row containing 1st symbol
    let mut rows: Vec<Row> = vec![Row {
        center: y_centers[0],
        height: usize::default(),
        sum: y_centers[0],
        symbols: vec![combined[0].clone()],
        superscript: None,
        subscript: None,
        parent: None,
    }];

    // Iterates across symbols and their centers (skipping 1st)
    for (y_center, symbol) in y_centers.into_iter().zip(combined.into_iter()).skip(1) {
        let mut new_row = true;
        // Iterate through existing rows checking if this symbols belongs to one
        for row in rows.iter_mut() {
            // If center of symbol is less than x% different, then it belongs to row. (x=100*ROW_CLEARANCE)
            if (1f32 - (y_center as f32 / row.center as f32)).abs() < ROW_CLEARANCE {
                row.symbols.push(symbol.clone());
                row.sum += y_center;
                row.center = row.sum / row.symbols.len();
                new_row = false; // Identifies a new row is not needed to contain said symbol
                break;
            }
        }
        // If symbol not put in existing row, create a new one.
        if new_row {
            rows.push(Row {
                center: y_center,
                height: usize::default(),
                sum: y_center,
                symbols: vec![symbol.clone()],
                superscript: None,
                subscript: None,
                parent: None,
            });
        }
    }

    // Prints symbols in rows
    #[cfg(debug_assertions)]
    {
        println!("rows (base):");
        for (indx, row) in rows.iter().enumerate() {
            println!("\t{} : {}", indx, row.print_symbols());
        }
    }

    // Construct composite symbols
    for row in rows.iter_mut() {
        let mut i = 0usize;
        // Can't use for loop since we use `.remove()` in loop (TODO Double check this)
        while i < row.symbols.len() {
            if row.symbols[i].class == "-" && i + 1 < row.symbols.len() {
                if row.symbols[i + 1].class == "-" {
                    // If difference between min x's is less than 20%
                    if (1f32
                        - row.symbols[i].bounds.min.x as f32
                            / row.symbols[i + 1].bounds.min.x as f32)
                        .abs()
                        <= 0.2f32
                    {
                        // Sets new symbol
                        row.symbols[i].class = "=".to_string(); // `=`
                                                                // Sets bounds
                        row.symbols[i].bounds =
                            Bound::from(&vec![&row.symbols[i].bounds, &row.symbols[i + 1].bounds]); // TODO How could I use slices here?
                                                                                                    // Removes component part
                        row.symbols.remove(i + 1);
                    }
                } else if i + 2 < row.symbols.len() {
                    // If `row.symbols[i+1]` and `row.symbols[i+2]` are contained within `row.symbols[i]`
                    if row.symbols[i + 1].class == "\\cdot "
                        && row.symbols[i + 2].class == "\\cdot "
                        && row.symbols[i]
                            .bounds
                            .contains_x(&[&row.symbols[i + 1].bounds, &row.symbols[i + 2].bounds])
                    {
                        // Sets new symbol
                        row.symbols[i].class = "\\div ".to_string(); // `\div`

                        // Calculate y bounds (which "." is on top and which is on bottom)
                        let (min_y, max_y) =
                            if row.symbols[i + 1].bounds.min.y < row.symbols[i + 2].bounds.min.y {
                                (
                                    row.symbols[i + 1].bounds.min.y,
                                    row.symbols[i + 2].bounds.max.y,
                                )
                            } else {
                                (
                                    row.symbols[i + 2].bounds.min.y,
                                    row.symbols[i + 1].bounds.max.y,
                                )
                            };
                        // Sets bounds
                        row.symbols[i].bounds = Bound {
                            min: Point {
                                x: row.symbols[i + 1].bounds.min.x,
                                y: min_y,
                            },
                            max: Point {
                                x: row.symbols[i + 1].bounds.max.x,
                                y: max_y,
                            },
                        };
                        // Removes component part
                        row.symbols.remove(i + 1);
                        row.symbols.remove(i + 1); // After first remove now i+1 == prev i+2
                    }
                }
            }
            i += 1;
        }
    }

    // Prints symbols in rows
    #[cfg(debug_assertions)]
    {
        println!("rows (combined symbols):");
        for (indx, row) in rows.iter().enumerate() {
            println!("\t{} : {}", indx, row.print_symbols());
        }
    }
    // Sorts rows in vertical order rows[0] is top row
    rows.sort_by_key(|r| r.center);

    // Prints symbols in rows and row centers
    #[cfg(debug_assertions)]
    {
        println!("rows (vertically ordered):");
        for (indx, row) in rows.iter().enumerate() {
            println!("\t{} : {}", indx, row.print_symbols());
        }
        let centers: Vec<usize> = rows.iter().map(|x| x.center).collect();
        println!("row centers: {:.?}", centers);
    }

    // Calculates average height of rows
    for row in rows.iter_mut() {
        let mut ignored_symbols = 0usize;
        for symbol in row.symbols.iter() {
            // Ignore the heights of '-' and '\\cdot' since there minuscule heights will throw off the average
            match symbol.class.as_str() {
                "-" | "\\cdot" => ignored_symbols += 1,
                _ => row.height += symbol.bounds.max.y - symbol.bounds.min.y, // `symbol.bounds.1.y - symbol.bounds.0.y` = height of symbol
            }
        }
        // Average height in row
        if row.symbols.len() != ignored_symbols {
            row.height /= row.symbols.len() - ignored_symbols;
        }
    }

    // Prints average row heights
    #[cfg(debug_assertions)]
    println!(
        "row heights: {:.?}",
        rows.iter().map(|x| x.height).collect::<Vec<usize>>()
    );

    // Contains references to rows not linked to another row as a sub/super script row
    // Initially contains a reference to every row.
    let mut unassigned_rows: Vec<&mut Row> = rows.iter_mut().collect();

    // Only 1 row is not a sub/super script row of another.
    // When we only have 1 unreferenced row we know we have linked all other rows as sub/super scripts.
    while unassigned_rows.len() > 1 {
        // List of indexes in reference to rows to remove from unassigned_rows as they have been assigned
        let mut removal_list: Vec<usize> = Vec::new();
        for i in 0..unassigned_rows.len() {
            let mut pos_sub = false; // Defines if this row could be a subscript row.
            if i > 0 {
                // If there is a row above this.
                // If the height of the row above is more than this, this could be a subscript to the row below
                if unassigned_rows[i - 1].height > unassigned_rows[i].height {
                    pos_sub = true;
                }
            }

            let mut pos_sup = false; // Defines if this row could be a superscript row.
            if i < unassigned_rows.len() - 1 {
                // If there is a row below this.
                // If the height of the row below is more than this, this could be a superscrit to the row below.
                if unassigned_rows[i + 1].height > unassigned_rows[i].height {
                    pos_sup = true;
                }
            }

            // Gets mutable raw pointer to this row
            let pointer: *mut Row = *unassigned_rows.get_mut(i).unwrap() as *mut Row;
            // If could both be superscript and subscript.
            // This row is a sub/super script to the row with smallest height
            if pos_sup && pos_sub {
                // If row below is smaller than row above, this row is a superscript to row below
                if unassigned_rows[i + 1].height < unassigned_rows[i - 1].height {
                    unassigned_rows[i + 1].superscript = Some(pointer); // Links parent to this as subscript
                    unassigned_rows[i].parent =
                        Some(*unassigned_rows.get_mut(i + 1).unwrap() as *mut Row); // Links to parent
                    removal_list.push(i);
                }
                // Else it is the subscript to the row above
                else {
                    unassigned_rows[i - 1].subscript = Some(pointer); // Links parent to this as superscript
                    unassigned_rows[i].parent =
                        Some(*unassigned_rows.get_mut(i - 1).unwrap() as *mut Row); // Links to parent
                    removal_list.push(i);
                }
            }
            // If could only be superscript
            else if pos_sub {
                unassigned_rows[i - 1].subscript = Some(pointer); // Links parent to this as superscript
                unassigned_rows[i].parent =
                    Some(*unassigned_rows.get_mut(i - 1).unwrap() as *mut Row); // Links to parent
                removal_list.push(i);
            }
            // If could only be subscript
            else if pos_sup {
                unassigned_rows[i + 1].superscript = Some(pointer); // Links parent to this as subscript
                unassigned_rows[i].parent =
                    Some(*unassigned_rows.get_mut(i + 1).unwrap() as *mut Row); // Links to parent
                removal_list.push(i);
            }
        }
        // Removes assigned rows from `unassigned_rows`
        remove_indexes(&mut unassigned_rows, &removal_list);
    }
    //println!("finished script setting");

    // Prints rows and linked rows (doesn't use `debug_out` since output is large, complex and interferes with later code. Not good for an overview)
    // unsafe {
    //     println!("\nrows:");
    //     for row in rows.iter() {
    //         println!();
    //         println!("{:.?}",row.print_symbols());
    //         if let Some(pointer) = row.subscript {
    //             println!("sub: {:.?} -> {:.?}",pointer,(*pointer).print_symbols())
    //         }
    //         if let Some(pointer) = row.superscript {
    //             println!("sup: {:.?} -> {:.?}",pointer,(*pointer).print_symbols())
    //         }
    //     }
    // }
    // return "".to_string();

    #[cfg(debug_assertions)]
    println!("{} : Scripts set", time(start));

    // The last remaining row in `unassigned_rows` must be the base row.

    // Sets 1st row
    let mut current_row: &mut Row = unassigned_rows.get_mut(0).unwrap();
    // Sets 1st symbol in latex
    let mut latex: String = current_row.symbols[0].class.clone();
    // Removes set symbol from row
    current_row.symbols.remove(0);
    unsafe {
        loop {
            #[cfg(debug_assertions)]
            println!("building: {}", latex);

            // TODO Make `min_sub` and `min_sup` immutable
            // Gets min x coordinate of next symbol in possible rows.
            //----------
            // Gets min x bound of symbol in subscript row
            let mut min_sub: usize = usize::max_value();
            if let Some(sub_row) = current_row.subscript {
                if let Some(symbol) = (*sub_row).symbols.first() {
                    min_sub = symbol.bounds.min.x;
                }
            }
            // Gets min x bound of symbol in superscript row
            let mut min_sup: usize = usize::max_value();
            if let Some(sup_row) = current_row.superscript {
                if let Some(symbol) = (*sup_row).symbols.first() {
                    min_sup = symbol.bounds.min.x;
                }
            }
            // Gets min x bound of next symbol in current row
            let min_cur: usize = if let Some(symbol) = current_row.symbols.get(0) {
                symbol.bounds.min.x
            } else {
                usize::max_value()
            };

            // Gets min x bounds of symbol in parent row
            let mut min_par: usize = usize::max_value();
            if let Some(parent) = current_row.parent {
                if let Some(symbol) = (*parent).symbols.first() {
                    min_par = symbol.bounds.min.x;
                }
            }

            //println!("(sub,sup,cur,par):({},{},{},{})",min_sub,min_sup,min_cur,min_par);

            // Finds minimum min x coordinate of next symbol in possible rows,
            //  switches to that row, appeneds that symbol to `latex` and removes that symbol from its row.
            //----------
            // ('closest' in this section means horizontally closest)
            if let Some(min) = min_option(&[min_sub, min_sup, min_cur, min_par]) {
                // If next closest symbol resides in the parent row, close this row and swtich to the parent row.
                if min == min_par {
                    current_row = &mut *current_row.parent.unwrap();
                    latex.push('}');
                }
                // If next closest symbol does not resides in the parent row
                else {
                    // If next closest symbol resides in subscript row, open subscript row, push 1st symbol and switch row.
                    if min == min_sub {
                        current_row = &mut *current_row.subscript.unwrap();
                        latex.push_str(&format!("_{{{}", current_row.symbols[0].class));
                    }
                    // If next closest symbol resides in superscript row, open subscript row, push 1st symbol and switch row.
                    else if min == min_sup {
                        current_row = &mut *current_row.superscript.unwrap();
                        latex.push_str(&format!("^{{{}", current_row.symbols[0].class));
                    }
                    // If next closest symbol resides in current row, push next symbol in current row.
                    else if min == min_cur {
                        latex.push_str(&current_row.symbols[0].class);
                    }
                    // Remove symbol added to latex
                    current_row.symbols.remove(0);
                }
            }
            // If next closest symbol not in parent, current, subscript or superscript row,
            else {
                // If there exists a parent row, close row and switch to parent row.
                if let Some(parent) = current_row.parent {
                    current_row = &mut *parent;
                    latex.push('}');
                }
                // If there does not exist a parent row, we are in the base row and at the end of the equation.
                else {
                    break;
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    println!("{} : Construction finished", time(start));

    return latex;

    // Returns some minimum from 4 element array, unless minium equals usize::max_value() then return none.
    fn min_option(slice: &[usize; 4]) -> Option<usize> {
        let min = *slice.iter().min().unwrap();
        if min == usize::max_value() {
            return None;
        }
        Some(min)
    }

    // Removes elements at given indices from given vector
    fn remove_indexes<T>(vec: &mut Vec<T>, indxs: &[usize]) {
        for (counter, indx) in indxs.iter().enumerate() {
            vec.remove(*indx - counter);
        }
    }
}

#[allow(dead_code)]
#[cfg(debug_assertions)]
fn time(instant: Instant) -> String {
    let millis = instant.elapsed().as_millis() % 1000;
    let seconds = instant.elapsed().as_secs() % 60;
    let mins = (instant.elapsed().as_secs() as f32 / 60f32).floor();
    format!("{:#02}:{:#02}:{:#03}", mins, seconds, millis)
}