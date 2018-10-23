import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMap;
import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem;

import com.google.protobuf.TextFormat;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import javax.imageio.ImageIO;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

public class DetectTable {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.exit(1);
        }
        final String[] labels = loadLabels(args[1]);
        try (SavedModelBundle model = SavedModelBundle.load(args[0], "serve")) {
            for (int arg = 2; arg < args.length; arg++) {
                final String filename = args[arg];
                List<Tensor<?>> outputs = null;
                try (Tensor<UInt8> input = makeImageTensor(filename)) {
                    outputs =
                            model
                                    .session()
                                    .runner()
                                    .feed("image_tensor", input)
                                    .fetch("detection_scores")
                                    .fetch("detection_classes")
                                    .fetch("detection_boxes")
                                    .run();
                }
                try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
                     Tensor<Float> classesT = outputs.get(1).expect(Float.class);
                     Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
                    int maxObjects = (int) scoresT.shape()[1];
                    float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
                    float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
                    float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
                    System.out.printf("* %s\n", filename);
                    boolean foundSomething = false;
                    for (int i = 0; i < scores.length; ++i) {
                        if (scores[i] < 0.8) {
                            continue;
                        }
                        foundSomething = true;
                        System.out.printf("\tFound %-20s (score: %.4f)\n", labels[(int) classes[i]], scores[i]);
                        System.out.println(boxes);
                    }
                    if (!foundSomething) {
                        System.out.println("No objects detected with a high enough score.");
                    }
                }
            }
        }
    }

    private static String[] loadLabels(String filename) throws Exception {
        String text = new String(Files.readAllBytes(Paths.get(filename)), StandardCharsets.UTF_8);
        StringIntLabelMap.Builder builder = StringIntLabelMap.newBuilder();
        TextFormat.merge(text, builder);
        StringIntLabelMap proto = builder.build();
        int maxId = 0;
        for (StringIntLabelMapItem item : proto.getItemList()) {
            if (item.getId() > maxId) {
                maxId = item.getId();
            }
        }
        String[] ret = new String[maxId + 1];
        for (StringIntLabelMapItem item : proto.getItemList()) {
            ret[item.getId()] = item.getDisplayName();
        }
        return ret;
    }

    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    private static Tensor<UInt8> makeImageTensor(String filename) throws IOException {
        BufferedImage img = ImageIO.read(new File(filename));
        if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new IOException(
                    String.format(
                            "Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
                            img.getType(), filename));
        }
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        bgr2rgb(data);
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[] {BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
    }

}