use crate::models::*;

use std::usize;

#[cfg(debug_assertions)]
use crate::util::time;
#[cfg(debug_assertions)]
use std::time::Instant;

const ROW_CLEARANCE: f32 = 0.3f32;

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
    #[cfg(debug_assertions)]
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
