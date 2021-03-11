use crate::models::*;

use std::{cmp, collections::VecDeque, fs, path::Path, usize};

use image::{imageops::FilterType, ImageBuffer, Luma};

// #[cfg(debug_assertions)]
use crate::util::time;
// #[cfg(debug_assertions)]
use num_format::{Locale, ToFormattedString};
// #[cfg(debug_assertions)]
use std::time::Instant;

const SCALE: u32 = 20u32;



// Returns symbol images and bounds
pub fn segment(
    path: &str,
    
    // Binarization parameters
    extreme_luma_boundary: u8,
    global_luma_boundary: u8,
    local_luma_boundary: u8,
    local_field_reach: i32,
    local_field_size: usize,
) -> Vec<Vec<(Vec<u8>,Bound<usize>)>> {
    // #[cfg(debug_assertions)]
    let start = Instant::now();

    // Open the image to segment (in this case it will reside within the `test_imgs` directory)
    let img = image::open(path).unwrap().to_luma8();

    // Gets dimensions of the image
    let dims = img.dimensions();
    let size = (dims.0 as usize, dims.1 as usize);

    // Gets raw pixel values from the image.
    let mut img_raw: Vec<u8> = img.into_raw();

    // 2d vector of size of image, where each pixel will be labelled white/black (and later used in forest fire)
    let mut pixels: Vec<Vec<Pixel>> = binarize(
        size,
        &mut img_raw,
        extreme_luma_boundary,
        global_luma_boundary,
        local_luma_boundary,
        local_field_reach,
        local_field_size
    );

    // Gets name of image file ('some_image.jpg' -> 'some_image')
    // #[cfg(debug_assertions)]
    let name = Path::new(path).file_stem().unwrap().to_str().unwrap();

    // Outputs binary image
    // #[cfg(debug_assertions)]
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
    let pixel_lists: Vec<Vec<(usize, usize)>> = get_pixel_groups(size, &mut pixels);

    // Gets bounds, square bounds and square bounds scaling property for each symbol
    let bounds: Vec<(Bound<usize>, (Bound<i32>, i32))> = get_bounds(&pixel_lists);

    // Outputs border image
    // #[cfg(debug_assertions)]
    output_bounds(2, [255, 0, 0], path, name, &img_raw, &bounds, size);

    // Gets lines in-between bounds
    let lines: Vec<Line> = get_lines(
        bounds.iter().map(|(b,_)|b).collect::<Vec<&Bound<usize>>>(),
        size.1
    );

    // #[cfg(debug_assertions)]
    println!("Lines: {:.?}", lines);

    // Outputs line image
    // #[cfg(debug_assertions)]
    output_lines([0, 0, 255],path,name,&img_raw,&lines,size.0);

    // Gets scaled pixels belonging to each symbol
    let symbols: Vec<Vec<u8>> = get_symbols(&pixel_lists, &bounds);

    // Outputs symbol images
    // #[cfg(debug_assertions)]
    output_symbols(&symbols, name);

    let symbols_lines: Vec<Vec<(Vec<u8>,Bound<usize>)>> = split_symbols_by_lines(
        symbols,
        bounds.into_iter().map(|(b,_)| b).collect(),
        lines
    );

    // #[cfg(debug_assertions)]
    println!("{} : Finished segmentation", time(start));

    symbols_lines
}

pub fn binarize(
    (width, height): (usize, usize),
    img_raw: &mut Vec<u8>,

    extreme_luma_boundary: u8,
    global_luma_boundary: u8,

    local_luma_boundary: u8,
    local_field_reach: i32,
    local_field_size: usize,
) -> Vec<Vec<Pixel>> {
    // #[cfg(debug_assertions)]
    let start = Instant::now();

    // Initializes 2d vector of size height*width with Pixel::White.
    let mut pixels: Vec<Vec<Pixel>> = vec![vec!(Pixel::White; width as usize); height as usize];

    // #[cfg(debug_assertions)]
    println!(
        "{} * {} = {}",
        width.to_formatted_string(&Locale::en),
        height.to_formatted_string(&Locale::en),
        img_raw.len().to_formatted_string(&Locale::en)
    );

    // Gets average luma among pixels
    //  Uses `.fold()` instead of `.sum()` as sum of values will likely exceed `u8:MAX`
    let global_luma: u8 =
        (img_raw.iter().fold(0u32, |sum, &x| sum + x as u32) / img_raw.len() as u32) as u8;
    let global_luma_sub = global_luma.checked_sub(global_luma_boundary);
    let global_luma_add = global_luma.checked_add(global_luma_boundary);

    // O(n*(2p)^2) // n = img_raw.len(), p = local_field_reach
    let rows = (height as f32 / local_field_size as f32).ceil() as usize;
    let cols = (width as f32 / local_field_size as f32).ceil() as usize;
    let chunks = rows * cols;

    // #[cfg(debug_assertions)]
    println!("{}:{}", rows, cols);
    // panic!("got here {}",chunks);

    let mut local_luma: Vec<u8> = vec![u8::default(); chunks];
    let mut chunk: usize = 0;

    for y in (0..height as i32).step_by(local_field_size) {
        let y_r = cmp::max(y - local_field_reach, 0)
            ..cmp::min(
                y + local_field_size as i32 + local_field_reach,
                height as i32,
            );
        let y_size = y_r.end - y_r.start;
        for x in (0..width as i32).step_by(local_field_size) {
            let x_r = cmp::max(x - local_field_reach, 0)
                ..cmp::min(
                    x + local_field_size as i32 + local_field_reach,
                    width as i32,
                );
            let x_size = x_r.end - x_r.start;

            let total: u32 = y_r
                .clone()
                .map(|yl| {
                    let start = (yl * width as i32 + x_r.start) as usize;
                    let end = start + x_size as usize;
                    img_raw[start..end].iter().map(|x| *x as u32).sum::<u32>()
                })
                .sum();

            //print!(" total:{}",total);
            //println!("{},{} \n {:.?},{:.?} \n {},{}",y_size,x_size,y_r,x_r,y,x);
            // if (y_size * x_size) == 0 || total / (y_size * x_size) as u16 > u8::max_value() as u16 {
            //     println!("{},{} \n {:.?},{:.?} \n {},{}",y_size,x_size,y_r,x_r,y,x);
            // }

            local_luma[chunk] = (total / (y_size * x_size) as u32) as u8;
            chunk += 1;
            //println!(" local_luma[i]:{}",local_luma[i]);
        }
    }

    // #[cfg(debug_assertions)]
    println!("{} : Calculated local luminance fields", time(start));

    // #[cfg(debug_assertions)]
    let start = Instant::now();

    for (y, row) in pixels.iter_mut().enumerate() {
        let y_group = y / local_field_size;
        for (x, p) in row.iter_mut().enumerate() {
            let x_group = x / local_field_size;
            let group = y_group * cols + x_group;

            let i = y * width + x; // Index of pixel of coordinates (x,y) in `img_raw`
            if img_raw[i] < extreme_luma_boundary {
                img_raw[i] = 0;
                *p = Pixel::Black;
                continue;
            }
            if img_raw[i] > 255 - extreme_luma_boundary {
                img_raw[i] = 255;
                // pixels `Pixel::White` by default
                continue;
            }
            if let Some(global_lower) = global_luma_sub {
                if img_raw[i] < global_lower {
                    img_raw[i] = 0;
                    *p = Pixel::Black;
                    continue;
                }
            }
            if let Some(global_upper) = global_luma_add {
                if img_raw[i] > global_upper {
                    img_raw[i] = 255;
                    // pixels `Pixel::White` by default
                    continue;
                }
            }
            if let Some(local) = local_luma[group].checked_sub(local_luma_boundary) {
                if img_raw[i] < local {
                    img_raw[i] = 0;
                    *p = Pixel::Black;
                    continue;
                }
            }
            if let Some(local) = local_luma[group].checked_add(local_luma_boundary) {
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

    // #[cfg(debug_assertions)]
    println!("{} : Converted image to binary", time(start));

    pixels
}

pub fn get_pixel_groups(
    (width, height): (usize, usize),
    pixels: &mut Vec<Vec<Pixel>>,
) -> Vec<Vec<(usize, usize)>> {
    // #[cfg(debug_assertions)]
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

    // #[cfg(debug_assertions)]
    println!("{} : Fill finished", time(start));

    // let min = width * height / 10000;
    // let pixel_lists: Vec<Vec<(usize, usize)>> = pixel_lists.into_iter().filter(|pl|pl.len() > min).collect();

    // Filter out shitty specs & return
    let min = width * height / 20000; // TODO Adapt to use size relative to largest characters
    return pixel_lists
        .into_iter()
        .filter(|pl| pl.len() > min)
        .collect();

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

pub fn get_bounds(pixel_lists: &[Vec<(usize, usize)>]) -> Vec<(Bound<usize>, (Bound<i32>, i32))> {
    // #[cfg(debug_assertions)]
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

    // #[cfg(debug_assertions)]
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

pub fn get_lines(bounds: Vec<&Bound<usize>>,height:usize) -> Vec<Line> {
    // #[cfg(debug_assertions)]
    let start = Instant::now();

    let mut lines = vec![
        Line { max: height, min: bounds[0].max.y },
        Line { max: bounds[0].min.y, min: 0 }
    ];
    let mut max_height = bounds[0].max.y - bounds[0].min.y;
    for b in bounds.into_iter().skip(1) {
        let h = b.max.y - b.min.y;
        if h > max_height { max_height = h; }
        for i in 0..lines.len() {
            // Writing these comparisons slightly more efficiently makes them ugly and hard to understand.

            // Covering
            if lines[i].max > b.max.y && lines[i].min < b.min.y {
                lines.push(Line { max: b.min.y, min: lines[i].min });
                lines[i].min = b.max.y;
            }
            // Upper intersection
            else if lines[i].max > b.max.y && lines[i].min < b.max.y {
                lines[i].min = b.max.y;
            }
            // Lower intersection
            else if lines[i].max > b.min.y && lines[i].min < b.min.y {
                lines[i].max = b.min.y;
            }
            // Covered
            else if lines[i].max < b.max.y && lines[i].min > b.min.y {
                lines.remove(i);
            }
        }
    }

    // #[cfg(debug_assertions)]
    println!("{} : Got lines", time(start));

    // Filter out all small lines (as they could be between super/sub script unless they are edge lines)
    lines.into_iter().filter(|l| (l.max - l.min) > (max_height / 4) || l.max == height || l.min == 0).collect()
}

// Returns tuple of: 2d vec of pixels in symbol, Bounds of symbol
pub fn get_symbols(
    pixel_lists: &[Vec<(usize, usize)>],
    bounds: &[(Bound<usize>, (Bound<i32>, i32))],
) -> Vec<Vec<u8>> {
    // #[cfg(debug_assertions)]
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
    // #[cfg(debug_assertions)]
    println!("{} : Symbols set", time(start));

    symbols
}

fn split_symbols_by_lines(mut symbols: Vec<Vec<u8>>, mut bounds: Vec<Bound<usize>>, mut lines: Vec<Line>) -> Vec<Vec<(Vec<u8>,Bound<usize>)>> {
    // #[cfg(debug_assertions)]
    let start = Instant::now();
    
    // Presuming already sorted, might need top come back to this if I'm wrong
    // TODO If this proves to always be true, switch back to stable from nightly
    lines.sort_by_key(|l|l.max);
    // #[cfg(debug_assertions)]
    println!("Sorted lines: {:.?}", lines);

     // The `-1` here could cause an error if there was not enough space to create lines at the bottom and top of an image.
    let mut symbol_lines: Vec<Vec<(Vec<u8>,Bound<usize>)>> = vec![Vec::new();lines.len()-1];
    // Assuming lines sorted by ascending max (from 0->max, given 0 is top, this is top to bottom technically)
    for (line, symbol_line) in lines.into_iter().skip(1).zip(symbol_lines.iter_mut()) {
        let mut adjust = 0;
        for i in 0..bounds.len() {
            if bounds[i-adjust].min.y < line.max {
                symbol_line.push((symbols.remove(i-adjust),bounds.remove(i-adjust)));
                adjust += 1;
            }
        }
    }

    // #[cfg(debug_assertions)]
    println!("Line widths: {:.?}", symbol_lines.iter().map(|s|s.len()).collect::<Vec<usize>>());

    // #[cfg(debug_assertions)]
    println!("{} : Split symbols into lines", time(start));

    symbol_lines
}

pub fn output_symbols(symbols: &[Vec<u8>], name: &str) {
    // #[cfg(debug_assertions)]
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
    // #[cfg(debug_assertions)]
    println!("{} : Symbols output", time(start));
}

pub fn output_bounds(
    spacing: usize,
    colour: [u8; 3],
    path: &str,
    name: &str,
    symbols: &[u8],
    bounds: &[(Bound<usize>, (Bound<i32>, i32))],
    (width, height): (usize, usize),
) {
    // #[cfg(debug_assertions)]
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

    // #[cfg(debug_assertions)]
    println!("{} : Bounds output", time(start));
}

pub fn output_lines(
    colour: [u8; 3],
    path: &str,
    name: &str,
    symbols: &[u8],
    lines: &[Line],
    width: usize,
) {
    // #[cfg(debug_assertions)]
    let start = Instant::now();

    // Opens another version of image
    let mut line_img = image::open(path).unwrap().into_rgb8();
    for (x, y, pixel) in line_img.enumerate_pixels_mut() {
        let val = symbols[((y as usize) * width) + (x as usize)];
        *pixel = image::Rgb([val, val, val]);
    }

    // Sets lines
    let line_pixel = image::Rgb(colour); // Pixel to use as line
    for l in lines.iter() {
        for i in 0..width {
            for t in l.min..l.max {
                *line_img.get_pixel_mut(i as u32, t as u32) = line_pixel;
            }
        }
    }
    if !Path::new("lines").exists() {
        fs::create_dir("lines").unwrap();
    }
    line_img.save(format!("lines/{}.png", name)).unwrap();

    // #[cfg(debug_assertions)]
    println!("{} : Lines output", time(start));
}
