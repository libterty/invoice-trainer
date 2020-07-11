import * as path from "path";
import * as cv from 'opencv4nodejs';

const _cv = cv;

const dataPath = path.resolve(__dirname, '../data');

declare type FrameHandler = (frame: cv.Mat) => unknown;

const grabFrames = (videoFile: string, delay: number, onFrame: FrameHandler) => {
  const cap: cv.VideoCapture = new _cv.VideoCapture(videoFile);
  let done: boolean = false;
  const intvl: NodeJS.Timeout = setInterval(() => {
    let frame: cv.Mat = cap.read();
    if (frame.empty) {
      cap.reset();
      frame = cap.read();
    }
    onFrame(frame);

    const key: number = _cv.waitKey(delay);
    done = key !== -1 && key !== 255;
    if (done) {
      clearInterval(intvl);
      console.log('Key pressed, exiting.');
    }
  }, 0);
};

const runVideoDetection = (src: string, detect: FrameHandler) => {
  grabFrames(src, 1, (frame: cv.Mat)  => {
    detect(frame);
  });
};

export {
  cv as _cv,
  dataPath,
  runVideoDetection
}