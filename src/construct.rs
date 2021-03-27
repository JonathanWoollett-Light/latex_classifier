use crate::models::*;

use std::{
    ops::{Sub, SubAssign},
    usize,
    fmt::{Debug,Display, Formatter ,Result},
    rc::Rc,cell::RefCell,
    collections::VecDeque
};
use phf::phf_set;

#[cfg(debug_assertions)]
use crate::util::time;
#[cfg(debug_assertions)]
use std::time::Instant;

const ROW_CLEARANCE: f32 = 0.3;
const EQUAL_CLEARANCE: f32 = 0.2;

// Symbols to ignore when calculating the average row heights
static IGNORE_SYMBOLS_ROW_HEIGHTS: phf::Set<&'static str> = phf_set! {
    "-","\\cdot"
};

// O(n^2 + 7n + 2n*log(n)) => O(n^2)
// n = number of symbols
pub fn construct(classes: &[&str], bounds: &[Bound<usize>]) -> String {
    #[cfg(debug_assertions)]
    let start = Instant::now();

    // Converts given symbols and bounds into `Symbol` structs
    // O(n)
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
    combined.sort_by_key(|a| ((a.bounds).min).j);

    // min x and y out of all symbols
    // O(1) - Min x
    let min_j: usize = combined[0].bounds.min.j;
    // O(n) - Min y
    let min_i: usize = bounds
        .iter()
        .min_by_key(|b| b.min.i)
        .expect("Bounds empty")
        .min
        .i;

    let origin = Point { i: min_i, j: min_j };

    // Subtract mins (`origin`) from bounds of all symbols
    // O(n)
    // TODO Which approach is better here?
    // let combined = combined.into_iter().map(|s| s - origin).collect();
    for symbol in combined.iter_mut() {
        *symbol -= origin;
    }

    // Initializes rows, 1st row containing 1st symbol
    let mut rows: Vec<Row> = vec![Row::new(combined.remove(0))];

    // Iterates across symbols and their centers (skipping 1st)
    // O(n^2 / 2)
    for symbol in combined.into_iter() {
        let i_center = symbol.center().i;

        // Iterate through existing rows checking if this symbols belongs to one
        let found_row = rows.iter().enumerate().find(|(_,r)| r.percentage_difference(i_center) < ROW_CLEARANCE);
        // If belongs to row, push to that row
        if let Some((i,_)) = found_row {
            rows[i].push(symbol,i_center);
        }
        // If doesn't belong to any row, push to new row
        else {
            rows.push(Row::new(symbol));
        }
    }

    // Prints symbols in rows
    #[cfg(debug_assertions)]
    {
        println!("rows (base):");
        for (indx, row) in rows.iter().enumerate() {
            println!("\t{}: {}", indx, row);
        }
    }

    // Construct composite symbols
    // O(n)
    compose_symbols(&mut rows);

    // Prints symbols in rows
    #[cfg(debug_assertions)]
    {
        println!("rows (combined):");
        for (indx, row) in rows.iter().enumerate() {
            println!("\t{}: {}", indx, row);
        }
    }

    // Sorts rows in descending vertical order (row[0] spatially top)
    // O(n log n)
    rows.sort_by_key(|r| r.center);

    // Prints symbols in rows
    #[cfg(debug_assertions)]
    {
        println!("rows (sorted):");
        for (indx, row) in rows.iter().enumerate() {
            println!("\t{}: {}", indx, row);
        }
    }

    // Calculates average height of rows
    // O(n)
    for row in rows.iter_mut() {
        row.set_height();
    }

    // Prints average row heights
    #[cfg(debug_assertions)]
    println!(
        "row heights: {:.?}",
        rows.iter().map(|r| r.height).collect::<Vec<usize>>()
    );

    #[cfg(debug_assertions)]
    println!("{} : Scripts set", time(start));

    // Set current row to base row (the row which is not a subscript or superscript of another)
    let base_row = script_rows(rows);

    #[cfg(debug_assertions)]
    println!("base row: {:.?}",base_row);

    // Sets 1st symbol in latex
    let mut latex: String = base_row.borrow_mut().symbols.pop_front().unwrap().class;
    latex.push_str(&traverse_scripts(base_row));

    #[cfg(debug_assertions)]
    println!("{} : Construction finished", time(start));

    return latex;
}
// O(n log n + n log n + n^2) -> O(n^2)
// n = number of rows
fn script_rows(rows: Vec<Row>) -> Rc<RefCell<Row>> {
    let mut lines: Vec<Rc<RefCell<Row>>> = rows.into_iter().map(|r|Rc::new(RefCell::new(r))).collect();
    lines.sort_by_key(|l|l.borrow().center);

    let mut line_stack: Vec<(usize,usize)> = lines.iter().enumerate().map(|(i,l)|(i,l.borrow().height)).collect();
    line_stack.sort_by_key(|(_,h)| *h );
    let base = Rc::clone(&lines[line_stack.last().unwrap().0]);

    // TODO the way we don't remove used entries but simply filter them is inefficient,
    //  a more efficient solution would not require filtering if a row is larger or assigned,
    //  we would simply remove such rows from the search space as the function progresses,
    //  this is fairly tricky, and I'm not sure how to do it, so for now this is why it
    //  hasn't been done, and tbf the maximum is like n=20, so O(n^2) kinda fine.
    while let Some((index,_)) = line_stack.pop() {
        let line_height = lines[index].borrow().height;

        // Lines below (lines.len()-1 is min height)
        if let Some(high_index) = (index+1..lines.len())
            .filter(|i| lines[*i].borrow().parent.is_none() && lines[*i].borrow().height < line_height)
            .max_by_key(|i| lines[*i].borrow().height)
        {
            lines[index].borrow_mut().subscript = Some(Rc::clone(&lines[high_index]));
            lines[high_index].borrow_mut().parent = Some(Rc::clone(&lines[index]));
        }
        

        // Lines above (0 max height)
        if index > 0 {
            if let Some(low_index) = (0..index)
                .filter(|i| lines[*i].borrow().parent.is_none() && lines[*i].borrow().height < line_height)
                .max_by_key(|i| lines[*i].borrow().height)
            {
                lines[index].borrow_mut().superscript = Some(Rc::clone(&lines[low_index]));
                lines[low_index].borrow_mut().parent = Some(Rc::clone(&lines[index]));
            }
        }   
    }
    base
}

fn percentage_difference(a:usize,b:usize) -> f32 {
    (a as f32 - b as f32).abs() / (((a+b) as f32) / 2.)
}

// Construct composite symbols
// O(n)
fn compose_symbols(rows: &mut Vec<Row>) {
    for row in rows.iter_mut() {
        let mut offset = 0;
        for i in 0..row.symbols.len() {
            let indx = i - offset;
            // If current symbol is "-" and this is not the last symbol
            if row.symbols[indx].class == "-" && indx < row.symbols.len() - 1 {
                // = route
                // If next symbol is "-" and min j's are close (within a percentage difference)
                if row.symbols[indx+1].class == "-" && 
                    percentage_difference(row.symbols[indx].bounds.min.j,row.symbols[indx + 1].bounds.min.j) < EQUAL_CLEARANCE
                {
                    row.symbols[indx].class = "=".to_string();
                    // Adds bounds, such that `row.symbols[i].bounds` covers both
                    let b = row.symbols[indx + 1].bounds; // This line neccessary to satisfy dumb borrow checker
                    row.symbols[indx].bounds += b;
                    // Remove component symbol
                    row.symbols.remove(indx + 1);
                    // Increment offset
                    offset += 1;
                }
                // /div route
                // If there are 2 following elements
                else if indx < row.symbols.len() - 2 {
                    // If both of the 2 following elements are \cdot and within the j bounds of the current - element
                    if row.symbols[indx + 1].class == "\\cdot" && 
                        row.symbols[indx + 2].class == "\\cdot" &&
                        row.symbols[indx].bounds.contains_j(&[&row.symbols[indx].bounds, &row.symbols[indx + 1].bounds]) 
                    {
                        row.symbols[indx].class = "\\div".to_string();
                        // Adds bounds, such that `row.symbols[i].bounds` covers all symbols
                        let b1 = row.symbols[indx + 1].bounds; // This line neccessary to satisfy dumb borrow checker
                        let b2 = row.symbols[indx + 2].bounds; // This line neccessary to satisfy dumb borrow checker
                        row.symbols[indx].bounds += b1 + b2;
                        // Remove component symbols
                        row.symbols.remove(indx + 1);
                        row.symbols.remove(indx + 1);
                        // Increment offset
                        offset += 2;
                    }
                }
            }
        }
    }
}

// O(n)
// n = total number of symbols across rows
fn traverse_scripts(row: Rc<RefCell<Row>>) -> String {
    let mut line = String::new();
    // Switch to superscript row
    { // This scope will constrain the lifetime of row_ref
        let row_ref = row.borrow();
        if let Some(superscript_row) = &row_ref.superscript {
            let mut child = superscript_row.borrow_mut();
            if let Some(superscript_symbol) = child.symbols.front() {
                if let Some(current_row_symbol) = row_ref.symbols.front() {
                    if superscript_symbol.before(current_row_symbol) {
                        start(&mut line,"^{ ",superscript_symbol);
                        child.symbols.pop_front();

                        drop(child); // child is no longer needed, so drop it before recursing
                        // Since superscript_row borrows from row_ref, we must Rc::clone it before
                        // dropping row_ref so that we can still pass it to traverse_scripts.
                        let superscript_row = Rc::clone(superscript_row);
                        drop(row_ref); // row_ref is no longer needed, so drop it before recursing
                        line.push_str(&traverse_scripts(superscript_row));
                        
                        end(&mut line);
                        
                    }
                } else {
                    start(&mut line,"^{ ",superscript_symbol);
                    child.symbols.pop_front();

                    drop(child);
                    let superscript_row = Rc::clone(superscript_row);
                    drop(row_ref);
                    line.push_str(&traverse_scripts(superscript_row));
                    
                    end(&mut line);
                }
            }
        } // child is dropped here (if it wasn't already). superscript_row is no longer borrowed
    } // row_ref is dropped here (if it wasn't already). row is no longer borrowed

    // Switch to subscript row
    {
        let row_ref = row.borrow();
        if let Some(subscript_row) = &row_ref.subscript {
            let mut child = subscript_row.borrow_mut();
            if let Some(subscript_symbol) = child.symbols.front() {
                if let Some(current_row_symbol) = row_ref.symbols.front() {
                    if subscript_symbol.before(current_row_symbol) {
                        start(&mut line,"_{ ",subscript_symbol);
                        child.symbols.pop_front();

                        drop(child);
                        let subscript_row = Rc::clone(subscript_row);
                        drop(row_ref);
                        line.push_str(&traverse_scripts(subscript_row));
                        
                        end(&mut line);
                    }
                } else {
                    start(&mut line,"_{ ",subscript_symbol);
                    child.symbols.pop_front();

                    drop(child);
                    let subscript_row = Rc::clone(subscript_row);
                    drop(row_ref);
                    line.push_str(&traverse_scripts(subscript_row));

                    end(&mut line);
                }
            }
        }
    }

    // Iterate through row
    {
        let mut row_mut = row.borrow_mut();
        if let Some(parent_row) = &row_mut.parent {
            let parent_ref = parent_row.borrow();
            if let Some(parent_symbol) = parent_ref.symbols.front() {
                let parent_symbol = parent_symbol.clone();
                drop(parent_ref);
                while let Some(current_row_symbol) = row_mut.symbols.front() {
                    // If next current row symbol is not before next parent symbol
                    if !current_row_symbol.before(&parent_symbol) { break; }

                    start(&mut line," ",current_row_symbol);
                    row_mut.symbols.pop_front();
                    
                    drop(row_mut);
                    
                    line.push_str(&traverse_scripts(Rc::clone(&row)));
                    row_mut =  row.borrow_mut();
                }
            } else {
                drop(parent_ref);
                while let Some(current_row_symbol) = row_mut.symbols.pop_front() {
                    start(&mut line," ",&current_row_symbol);

                    drop(row_mut);
                    line.push_str(&traverse_scripts(Rc::clone(&row)));
                    row_mut =  row.borrow_mut();
                }
            }
        }
        else {
            while let Some(current_row_symbol) = row_mut.symbols.pop_front() {
                start(&mut line," ",&current_row_symbol);

                drop(row_mut);
                line.push_str(&traverse_scripts(Rc::clone(&row)));
                row_mut =  row.borrow_mut();
            }
        }
    }

    // #[cfg(debug_assertions)]
    // println!("\t{}",line);

    return line;

    fn start<T: std::fmt::Display>(line: &mut String, prefix: &str, char: &T) {
        let start = format!("{}{}",prefix,char);
        #[cfg(debug_assertions)]
        print!("{}",start);
        line.push_str(&start);
    }

    fn end(line: &mut String) {
        let end = String::from(" }");
        #[cfg(debug_assertions)]
        print!("{}",end);
        line.push_str(&end);
    }
}

// Struct for symbol
#[derive(Clone, Debug)]
struct Symbol {
    class: String,
    bounds: Bound<usize>,
}
impl Symbol {
    fn center(&self) -> Point<usize> {
        self.bounds.center()
    }
    // i_size of bounds
    fn height(&self) -> usize {
        self.bounds.max.i - self.bounds.min.i
    }
    // if min.j of self is less than min.j of other (i,e. if self is before other on the x axis)
    fn before(&self, other: &Symbol) -> bool {
        self.bounds.min.j < other.bounds.min.j
    } 
}
impl SubAssign<Point<usize>> for Symbol {
    fn sub_assign(&mut self, other: Point<usize>) {
        self.bounds -= other;
    }
}
impl Sub<Point<usize>> for Symbol {
    type Output = Self;
    fn sub(self, other: Point<usize>) -> Self {
        Self {
            class: self.class,
            bounds: self.bounds - other
        }
    }
}
impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f,"{}",self.class)
    }
}

// Struct for row
struct Row {
    center: usize,
    height: usize,
    sum: usize,
    symbols: VecDeque<Symbol>,
    superscript: Option<Rc<RefCell<Row>>>,
    subscript: Option<Rc<RefCell<Row>>>,
    parent: Option<Rc<RefCell<Row>>>,
}
#[cfg(debug_assertions)]
impl Row {
    fn new(symbol: Symbol) -> Self {
        let center = symbol.center().i;
        Row {
            center: center,
            height: usize::default(),
            sum: center,
            symbols: VecDeque::from(vec![symbol]),
            superscript: None,
            subscript: None,
            parent: None,
        }
    }
    fn push(&mut self, symbol: Symbol, center: usize) {
        self.symbols.push_back(symbol);
        self.sum += center;
        self.center = self.sum / self.symbols.len();
    }
    // Percentage difference between row center and given value
    fn percentage_difference(&self, other: usize) -> f32 {
        percentage_difference(self.center,other)
    }
    // Calculates height
    fn set_height(&mut self) {
        let set: Vec<usize> = self.symbols.iter().filter_map(|s| 
            if IGNORE_SYMBOLS_ROW_HEIGHTS.contains(&s.class as &str) { None } else { Some(s.height()) }
        ).collect();
        self.height = set.iter().sum::<usize>() / set.len();
    }
}
impl Display for Row {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let out = format!(
            "[ {}]",
            self.symbols
                .iter()
                .map(|s| format!("{} ", s.class))
                .collect::<String>()
        );
        // let mut list: String = self.symbols.iter().map(|s| write!(f,"{}",s)).collect();
        write!(f,"{}",out)
    }
}
impl Debug for Row {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.debug_struct("Row")
        .field("symbols",&self.height)
        .field("superscript",&self.superscript)
        .field("subscript",&self.subscript)   
        .finish()
    }
}