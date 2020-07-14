import * as path from 'path';
import * as fs from 'fs';
import Jimp from 'jimp';
import { TensorFlowInception } from './dnn/tensorflow.inception';
import { _cv } from './utils/index';
const tensorFlowInception = new TensorFlowInception();

interface IImage {
  image: string;
  label: string
}

async function main(): Promise<void> {
  const images: IImage[] = [
    {
      image: path.resolve(__dirname, '../data/images/inv.jpeg'),
      label: 'invoice1'
    },
    {
      image: path.resolve(__dirname, '../data/images/inv2.jpeg'),
      label: 'invoice2'
    }
  ];

  for (let i = 0; i < images.length; i++) {
    await Jimp.read(images[i].image)
      .then((lenna: Jimp) => {
        return lenna
          .resize(255, 255) // resize
          .quality(60) // set JPEG quality
          .greyscale() // set greyscale
          .write(`./data/resizes/${images[i].label}_resize.jpg`); // save
      })
      .catch(err => {
        console.error(err);
      });
  }

  const resize_images: string[] = fs.readdirSync(path.resolve(__dirname, '../data/resizes'));
  
  for (let i = 0; i < resize_images.length; i++) {
    const image_path: string = path.resolve(__dirname, `../data/resizes/${resize_images[i]}`);
    // const img: _cv.Net = _cv.imread(image_path);
    const predictions = tensorFlowInception.classifyImage(image_path);
    console.log('pre', predictions)
  }
}

main();