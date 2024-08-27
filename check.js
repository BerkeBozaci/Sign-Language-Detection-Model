async function predict(inputData) {
  //console.log(inputData);
  const model = await loadModel();

  // Assuming inputData is a JavaScript array that matches the input shape expected by your model
  const tensor = tf.tensor(inputData).reshape([1, inputData.length]); // Adjust shape if necessary

  // Perform prediction
  const prediction = model.predict(tensor);

  // Output the prediction
  //prediction.print(); // For debugging, prints the tensor
  const predictedLabel = prediction.argMax(-1).dataSync()[0];
  //console.log(`Predicted Label: ${predictedLabel}`);

  // Convert the predicted label to the corresponding ASL letter
  const labelToLetter = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
  ];
  const predictedLetter = labelToLetter[predictedLabel];
  console.log(`Predicted Letter: ${predictedLetter}`);

  return predictedLetter;
}

async function loadModel() {
  const model = await tf.loadGraphModel("/model_tfjs/model.json");
  return model;
}

// Mock Data Examples for Testing
const mockDataList = [
  [1, 2, 0, 0, 0, 1, 0, 4, 0, 4], // M
  [1, 1, 2, 3, 0, 4, 0, 4, 0, 4], // L
  [1, 1, 0, 3, 0, 4, 0, 1, 2, 1], // Y
  [1, 2, 0, 0, 0, 1, 0, 0, 0, 4], // N
  [1, 1, 0, 0, 0, 1, 1, 0, 1, 0], // O
  [1, 4, 2, 1, 0, 2, 0, 3, 0, 3], // X
  [0, 2, 0, 0, 0, 4, 0, 4, 2, 4], // I
  [1, 4, 2, 2, 2, 3, 0, 3, 0, 3], // H
  [1, 5, 2, 3, 1, 4, 0, 4, 0, 4], // K
  [0, 2, 2, 3, 2, 4, 2, 4, 1, 4], // W
  [1, 1, 0, 0, 0, 4, 0, 4, 0, 4], // A
  [1, 2, 2, 3, 2, 4, 0, 0, 0, 4], // V
  [1, 5, 0, 0, 0, 4, 0, 4, 0, 4], // T
  [1, 5, 2, 3, 2, 4, 2, 4, 2, 4], // B
  [1, 3, 1, 0, 1, 1, 1, 0, 1, 0], // C
  [0, 2, 2, 3, 2, 4, 0, 4, 0, 4], // U
  [1, 1, 1, 0, 2, 4, 2, 4, 2, 4], // F
  [1, 4, 1, 2, 0, 2, 0, 3, 0, 3], // Q
  [1, 2, 2, 1, 1, 3, 0, 3, 0, 3], // P
  [1, 4, 2, 1, 0, 2, 0, 1, 0, 1], // G
  [0, 2, 0, 3, 0, 4, 0, 4, 0, 1], // E
  [1, 2, 2, 3, 2, 4, 0, 4, 0, 0], // R
  [0, 5, 0, 3, 0, 4, 0, 4, 0, 1], // S
  [0, 5, 2, 3, 1, 4, 1, 4, 1, 1], // D
];

mockDataList.forEach(async (mockData) => {
  const predictedLetter = await predict(mockData);
  console.log(
    `Input: ${mockData.join(",")} -> Predicted ASL Letter: ${predictedLetter}`
  );
});
