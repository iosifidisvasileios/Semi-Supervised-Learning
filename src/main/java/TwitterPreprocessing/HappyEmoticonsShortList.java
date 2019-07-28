package TwitterPreprocessing;

import java.util.HashSet;

/**
 * Created by iosifidis on 06.08.16.
 */
public class HappyEmoticonsShortList {
    private HashSet<String> happyArray = new HashSet<>();

    public HappyEmoticonsShortList() {
        happyArray.add(":)");
        happyArray.add(":-)");
        happyArray.add(":D");
        happyArray.add(";)");
        happyArray.add(":]");
        happyArray.add(":P");
        happyArray.add(":3");
        happyArray.add(":o)");
        happyArray.add(":c)");
        happyArray.add(":>)");
        happyArray.add("=]");
        happyArray.add("8)");
        happyArray.add("=)");
        happyArray.add(":}");
        happyArray.add(":^)");
        happyArray.add("<3");
        happyArray.add("^_^");
        happyArray.add(";>");
        happyArray.add("(:");
        happyArray.add("(;");
    }

    public HashSet<String> getHappyArray() {
        return happyArray;
    }
}
