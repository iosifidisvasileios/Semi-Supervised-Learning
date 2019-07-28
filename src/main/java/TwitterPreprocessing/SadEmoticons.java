package TwitterPreprocessing;

import java.util.HashSet;

/**
 * Created by iosifidis on 06.08.16.
 */
public class SadEmoticons {
    private HashSet<String> sadArray = new HashSet<>();

    public SadEmoticons() {
        sadArray.add(">:]");
        sadArray.add(":-(");
        sadArray.add(":(");
        sadArray.add(":-c");
        sadArray.add(":c");
        sadArray.add(":-<");
        sadArray.add(":<");
        sadArray.add(":-[");
        sadArray.add(":[");
        sadArray.add(":{");
        sadArray.add(";(");
        sadArray.add(":-||");
        sadArray.add(":@");
        sadArray.add(">:(");
        sadArray.add(":'-(");
        sadArray.add(":'(");
        sadArray.add("D:<");
        sadArray.add("D8");
        sadArray.add("D;");
        sadArray.add("D=");
        sadArray.add("DX");
        sadArray.add("v.v");
        sadArray.add("D‑':");
        sadArray.add(">:O");
        sadArray.add(":-O");
        sadArray.add(":O");
        sadArray.add(":-o");
        sadArray.add(":o");
        sadArray.add("8‑0");
        sadArray.add("O_O");
        sadArray.add("o‑o");
        sadArray.add("O_o");
        sadArray.add("o_O");
        sadArray.add("o_o");
        sadArray.add("O-O");
        sadArray.add(">:/");
        sadArray.add("=/");
        sadArray.add("T_T");
        sadArray.add("=(");
        sadArray.add("=[");
        sadArray.add("😷");
        sadArray.add("😵");
        sadArray.add("😰");
        sadArray.add("😯");
        sadArray.add("😮");
        sadArray.add("😳");
        sadArray.add("😱");
        sadArray.add("😤");
        sadArray.add("😨");
        sadArray.add("😧");
        sadArray.add("😦");
        sadArray.add("😥");
        sadArray.add("😬");
        sadArray.add("😫");
        sadArray.add("😩");
        sadArray.add("😠");
        sadArray.add("😟");
        sadArray.add("😞");
        sadArray.add("😣");
        sadArray.add("😢");
        sadArray.add("😡");
        sadArray.add("🙀");
        sadArray.add("😿");
        sadArray.add("😾");
        sadArray.add("😖");
        sadArray.add("😕");
        sadArray.add("😔");
        sadArray.add("😓");
        sadArray.add("😒");
        sadArray.add("👿");

//        sadArray.add("U+1F62");
    }

    public HashSet<String> getSadArray() {
        return sadArray;
    }
}
