package TwitterPreprocessing;

/**
 * Created by iosifidis on 07.08.16.
 */
public class EmoticonResult {
    int counter = 0;
    String emoticons = "";

    public int getCounter() {
        return counter;
    }

    public void increaseCounter() {
        this.counter ++;
    }


    public String getEmoticons() {
        return this.emoticons;
    }

    public void setEmoticons(String emoticons) {
        this.emoticons += emoticons + " ";
    }
}
