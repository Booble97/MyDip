public List<Rect> detectText(Mat rgbImg)
{
    isDetecting = true;

    Log.i(TAG, "Text detection...");

    // очистить список прямоугольников перед новым обнаружением текста.
    if(rectangles != null && rectangles.size() > 0)
    {
        rectangles.clear();
    }

    large = rgbImg;
    rgb = new Mat();

    // даунсэмпл для дальнейшего использования
    // По умолчанию вычисляется так: Size((src.cols+1)/2, (src.rows+1)/2)
    Imgproc.pyrDown(large, rgb);

    small = new Mat();

    Imgproc.cvtColor(rgb, small, Imgproc.COLOR_BGR2GRAY);

    // морфологический градиент
    grad = new Mat();
    Mat morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
    Imgproc.morphologyEx(small, grad, Imgproc.MORPH_GRADIENT, morphKernel);

    // бинаризация
    bw = new Mat();
    Imgproc.threshold(grad, bw, 0.0, 255.0, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

    // соединяент горизонтально ориентированные регионы
    connected = new Mat();
    morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9, 1));
    Imgproc.morphologyEx(bw, connected, Imgproc.MORPH_CLOSE, morphKernel);

    // находит контуры
    mask = Mat.zeros(bw.size(), CvType.CV_8UC1);
    contours = new ArrayList<MatOfPoint>();
    hierarchy = new MatOfInt4();

    Imgproc.findContours(connected, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

    // фильтрует контуры
    if (hierarchy != null && hierarchy.size().height > 0 && hierarchy.size().width > 0)
    {
        for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
        {
            Rect rect = Imgproc.boundingRect(contours.get(idx));

            Mat maskROI = new Mat(mask, rect);
            maskROI.setTo(new Scalar(0, 0, 0));

            // заполняет контуры
            Imgproc.drawContours(mask, contours, idx, new Scalar(255, 255, 255), Core.FILLED);

            // находит соотношение ненулевых пикселей в заполненном регионе
            double r = (double) countNonZero(maskROI) / (rect.width * rect.height);

            if (r > .5 &&                              // предпологаем, что по-крайней мере половина заполненного региона заполнена, если содержит текст
                (rect.height > 15 && rect.width > 30)) // ограничивает размер региона
            {
                rectangles.add(new Rect(rect.x * downsampleRatio,
                                        rect.y * downsampleRatio,
                                        rect.width * downsampleRatio,
                                        rect.height * downsampleRatio));
            }
        }
    }

    rectanglesToDraw.clear();
    rectanglesToDraw.addAll(rectangles);

    Log.i(TAG, rectangles.size() + " segments detected.");
    Log.i(TAG, "Text detection complete");

    isDetecting = false;

    Imgproc.pyrUp(rgb, rgb, new Size(1280, 720));

    return rectangles;
}
