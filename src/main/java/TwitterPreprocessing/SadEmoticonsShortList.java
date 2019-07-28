package TwitterPreprocessing;

import java.util.HashSet;

/**
 * Created by iosifidis on 06.08.16.
 */
public class SadEmoticonsShortList {
    private HashSet<String> sadArray = new HashSet<>();

    public SadEmoticonsShortList() {
        sadArray.add(":-(");
        sadArray.add(":(");
        sadArray.add(":-c");
        sadArray.add(":c");
        sadArray.add(":-<");
        sadArray.add(":<");
        sadArray.add(">:[");
        sadArray.add(":[");
        sadArray.add(":-[");
        sadArray.add(":{");
        sadArray.add(":'{");
        sadArray.add(":/");
    }

    public HashSet<String> getSadArray() {
        return sadArray;
    }
}
