use std::{
    f32::consts::PI,
    io::{self, Write},
};

use macroquad::prelude::*;
use math::{matrix, Matrix};
use sim::Network;

fn draw_network(network: &Network) {
    let x_offset = 60.0;
    let y_offset = 60.0;
    let x_spacing = 200.0;
    let y_spacing = 100.0;

    for (i, weights) in network.weights.iter().enumerate() {
        for source in 0..weights.rows {
            for dest in 0..weights.cols {
                let weight = weights.data[source * weights.cols + dest];
                let (line_colour, brightness) = if weight > 0.0 {
                    let brightness = weight / (weight + 5.0);
                    let partial = 255 - (255.0 * brightness) as u8;
                    (Color::from_rgba(255, partial, partial, 255), brightness)
                } else {
                    let brightness = -weight / (-weight + 5.0);
                    let partial = 255 - (255.0 * brightness) as u8;
                    (Color::from_rgba(partial, partial, 255, 255), brightness)
                };
                draw_line(
                    x_offset + x_spacing * i as f32,
                    y_offset + y_spacing * source as f32,
                    x_offset + x_spacing * (i + 1) as f32,
                    y_offset + y_spacing * dest as f32,
                    20.0 * brightness,
                    line_colour,
                )
            }
        }
    }

    for (i, layer_size) in network.layers.iter().enumerate() {
        for j in 0..*layer_size {
            draw_circle(
                x_offset + x_spacing * i as f32,
                y_offset + y_spacing * j as f32,
                10.0,
                WHITE,
            )
        }
    }
}

fn draw_graph(points: &[f32]) {
    let x_offset = 1000.0;
    let y_offset = 60.0;
    let width = 200.0;
    let height = 200.0;
    let xrange = 20000.0;
    let yrange = points.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    draw_rectangle_lines(x_offset, y_offset, width, height, 3.0, WHITE);
    for (i, points) in points.windows(2).enumerate() {
        let p1 = points[0];
        let p2 = points[1];
        let p1_y = -p1 / yrange * height + y_offset + height;
        let p2_y = -p2 / yrange * height + y_offset + height;
        let p1_x = (i as f32 * 20.0) / xrange * width + x_offset;
        let p2_x = ((i + 1) as f32 * 20.0) / xrange * width + x_offset;
        draw_line(p1_x, p1_y, p2_x, p2_y, 3.0, RED);
    }
}

fn input() -> String {
    print!(" Input a number in radians > ");
    io::stdout().flush().unwrap();
    let mut temp = String::new();
    io::stdin().read_line(&mut temp).unwrap();
    temp.trim().to_string()
}

#[macroquad::main("Neurs")]
async fn main() {
    // let x_data = vec![
    //     matrix![0.0, 0.0],
    //     matrix![0.0, 1.0],
    //     matrix![1.0, 0.0],
    //     matrix![1.0, 1.0],
    // ];
    // let y_data = vec![matrix![0.0], matrix![1.0], matrix![1.0], matrix![0.0]];

    let mut x_data = vec![];
    let mut y_data = vec![];

    let iters = 3000;
    for i in 0..iters {
        let frac = i as f32 / iters as f32;
        let value = frac * (PI * 2.0);
        let x = frac;
        let y = (value.sin() + 1.0) / 2.0;
        x_data.push(matrix![x]);
        y_data.push(matrix![y]);
    }

    println!("{:#?} {:#?}", x_data, y_data);

    let mut x_clone = vec![];
    for mut item in x_data.clone() {
        item.add_col(1.0);
        x_clone.push(item);
    }

    let mut network = Network::new(&[1, 3, 3, 1], 0.2);
    let mut loss_y = vec![];

    for i in 0..5000 {
        clear_background(BLACK);
        draw_network(&network);
        network.fit(&x_data, &y_data, 1, 30);
        let loss = network.calculate_loss(&x_clone, &y_data);
        println!("[INFO] |{}| current loss: {:?}", i, loss);
        loss_y.push(loss);
        draw_graph(&loss_y);
        next_frame().await
    }

    loop {
        let inp = input().parse::<f32>().unwrap() / (PI * 2.0);
        let predict = network.predict(&matrix![inp, 1.0]).data[0] * 2.0 - 1.0;
        println!("Ouptut: {}", predict);
    }
}
