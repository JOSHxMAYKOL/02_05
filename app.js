let model;
let webcam;
let isPredicting = false;

const labels = ['celular', 'mouse', 'teclado', 'lentes', 'no se reconoce nada'];
const MODEL_URL = 'model/model.json';

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const video = document.getElementById('webcam');
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.addEventListener('loadeddata', () => resolve(video));
      })
      .catch(err => reject(err));
  });
}

async function loadModel() {
  model = await tf.loadLayersModel(MODEL_URL);
}

async function predict() {
  const video = document.getElementById('webcam');
  const resultado = document.getElementById('resultado');

  while (isPredicting) {
    const img = tf.browser.fromPixels(video);
    const resized = tf.image.resizeBilinear(img, [224, 224]);
    const normalized = resized.div(255).expandDims(0);

    const prediction = await model.predict(normalized).data();
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    
    resultado.innerText = `Objeto: ${labels[maxIndex]} \nConfianza: ${(prediction[maxIndex] * 100).toFixed(2)}%`;

    img.dispose();
    resized.dispose();
    normalized.dispose();

    await tf.nextFrame();
  }
}

document.getElementById('start').addEventListener('click', async () => {
  if (!model) await loadModel();
  if (!webcam) webcam = await setupWebcam();
  isPredicting = true;
  predict();
});
