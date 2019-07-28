package TwitterPreprocessing;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.log4j.Logger;

import java.io.IOException;

/**
 * Created by iosifidis on 14.08.16.
 */
public class TaggerLoader {
    private MaxentTagger tagger;
    static Logger logger = Logger.getLogger(TaggerLoader.class);


    public TaggerLoader() {

        FileSystem fs = null;
        try {
            fs = FileSystem.get(new Configuration());
            tagger = new MaxentTagger("english-bidirectional-distsim.tagger");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public MaxentTagger getTagger() {
        return tagger;
    }
}
