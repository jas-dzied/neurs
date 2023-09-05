use math::{matrix, Matrix};
use sim::Network;

fn main() {
    let x_data = vec![
        matrix![0.0, 0.0],
        matrix![0.0, 1.0],
        matrix![1.0, 0.0],
        matrix![1.0, 1.0],
    ];

    let y_data = vec![matrix![0.0], matrix![1.0], matrix![1.0], matrix![0.0]];

    let mut network = Network::new(&[2, 2, 1], 0.5);
    network.fit(x_data.clone(), y_data.clone(), 200000, 10000);

    for (input, _) in x_data.iter().zip(y_data.iter()) {
        let mut input = input.clone();
        input.add_col(1.0);
        let prediction = network.predict(&input);
        println!(
            "[RESULT] input={:?}, prediction={:?}",
            input.repr(),
            prediction.repr()
        );
    }
}
