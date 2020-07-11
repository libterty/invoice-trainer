import { _cv } from '../utils/index';
import * as fs from 'fs';
import * as path from 'path';

export class TensorFlowInception {
  public inceptionModelPath: string;
  public modelFile: string;
  public classNamesFile: string;
  public classNames: string[];
  public net: _cv.Net;

  constructor() {
    this.inceptionModelPath = '../data/dnn/tf-inception';
    this.modelFile = path.resolve(this.inceptionModelPath, 'tensorflow_inception_graph.pb');
    this.classNamesFile = path.resolve(this.inceptionModelPath, 'imagenet_comp_graph_label_strings.txt');

    this.init();

    this.classNames = fs.readFileSync(this.classNamesFile).toString().split('\n');
    this.net = _cv.readNetFromTensorflow(this.modelFile);
  }

  private init(): void {
    if (!fs.existsSync(this.modelFile) || !fs.existsSync(this.classNamesFile)) {
      throw new Error('could not find inception model');
    }
  }

  /**
   * classifyImage
   */
  public classifyImage(img: any): string[] {
    // inception model works with 224 x 224 images, so we resize
    // our input images and pad the image with white pixels to
    // make the images have the same width and height
    const maxImgDim: number = 224;
    const white: _cv.Vec3 = new _cv.Vec3(255, 255, 255);
    const imgResized: _cv.Mat = _cv.imread(img).resizeToMax(maxImgDim).padToSquare(white);

    // network accepts blobs as input
    const inputBlob: _cv.Mat = _cv.blobFromImage(imgResized);
    this.net.setInput(inputBlob);

    // forward pass input through entire network, will return
    // classification result as 1xN Mat with confidences of each class
    const outputBlob: _cv.Mat = this.net.forward();

    // find all labels with a minimum confidence
    const minConfidence: number = 0.05;
    const locations: _cv.Point2[] =
      outputBlob
        .threshold(minConfidence, 1, _cv.THRESH_BINARY)
        .convertTo(_cv.CV_8U)
        .findNonZero();

    const result: string[] =
      locations.map(pt => ({
        confidence: parseInt((outputBlob.at(0, 3) * 100).toString()) / 100,
        className: this.classNames[pt.x]
      }))
        // sort result by confidence
        .sort((r0, r1) => r1.confidence - r0.confidence)
        .map(res => `${res.className} (${res.confidence})`);

    return result;
  }
}