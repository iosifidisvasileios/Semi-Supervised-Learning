package TwitterPreprocessing; /**
 * Created by iosifidis on 01.08.16.
 */

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.IOException;

public final class PreprocessText {

    private static final ProcessingOfRow processingOfRow = new ProcessingOfRow();
    static Logger logger = Logger.getLogger(PreprocessText.class);

    private static class MyMapper extends Mapper<LongWritable, Text, Text, Text> {
        protected void map(LongWritable key, Text row, Context context) throws IOException, InterruptedException {

            final String items = row.toString();

            try {
                String preprocessed = "";
                String outpout = "";

                String[] stringArray = items.split("\t");

//                final String idKey = String.valueOf(Long.valueOf(stringArray[0].split("\t")[0]));
                final String idKey = String.valueOf(Long.valueOf(stringArray[0]));

                preprocessed = stringArray[2];
                outpout = stringArray[1]  + "\t" + stringArray[2];// + "," + stringArray[3]+ "," + stringArray[4];

                //before deletion of some characters do a slang check
                preprocessed = processingOfRow.slangProcess(preprocessed.toLowerCase());

                preprocessed = processingOfRow.clearText(preprocessed);
                preprocessed = processingOfRow.clearWhitespace(preprocessed);

                //lowercase words, remove repetitions, replace slang terms
                preprocessed = processingOfRow.slangProcess(preprocessed);
                preprocessed = processingOfRow.clearWhitespace(preprocessed);

                //count emoticons, store them into string
                final EmoticonResult happyResult = processingOfRow.happyCounting(preprocessed);
                final EmoticonResult sadResult = processingOfRow.sadCounting(preprocessed);

                //count emoticons and store them into string
                preprocessed = processingOfRow.negationsBasedOnVerbs(preprocessed);
                preprocessed = processingOfRow.clearWhitespace(preprocessed);

                preprocessed = processingOfRow.negationsBasedOnAdj(preprocessed);
                preprocessed = processingOfRow.clearWhitespace(preprocessed);

                //remove emoticons, non ascii chars, symbols
                preprocessed = processingOfRow.removeNonAscii(preprocessed);
                preprocessed = processingOfRow.clearWhitespace(preprocessed);

                preprocessed = processingOfRow.removeStopwords(preprocessed);
                preprocessed = processingOfRow.clearWhitespace(preprocessed);

                String emotion ="";
                String emoticonString="";

                if (happyResult.getCounter() == 0 && sadResult.getCounter() == 0){
                    emotion = "neutral";
                    emoticonString = "null";

                }else if(happyResult.getCounter() > 0 && sadResult.getCounter() == 0){
                    emotion = "positive";
                    emoticonString = happyResult.getEmoticons();

                }else if(sadResult.getCounter() > 0 && happyResult.getCounter() == 0){
                    emotion = "negative";
                    emoticonString = sadResult.getEmoticons();

                }else if(happyResult.getCounter() > 0 && happyResult.getCounter() > sadResult.getCounter()){
                    emotion = "semi_positive";
                    emoticonString = happyResult.getEmoticons() + sadResult.getEmoticons();

                }else if(sadResult.getCounter() > 0 && sadResult.getCounter()> happyResult.getCounter()){
                    emotion = "semi_negative";
                    emoticonString = happyResult.getEmoticons() + sadResult.getEmoticons();

                }else{
                    emotion = "unknown";
                    emoticonString = happyResult.getEmoticons() + sadResult.getEmoticons();
                }
                outpout += "\t" + emotion + "\t" + emoticonString + "\t" + preprocessed;
                if ((emotion.equals("positive") || emotion.equals("negative") || emotion.equals("neutral"))) {
                    context.write(new Text(idKey), new Text(outpout));
                }
            }catch(NumberFormatException e){
                logger.error(e);
//                context.write(new Text("NumberFormatException"),new Text("1"));
//                cnt += 1;
            }catch (ArrayIndexOutOfBoundsException e){
                logger.error(e);
//                context.write(new Text("ArrayIndexOutOfBoundsException"),new Text("1"));
//                cnt2 += 1;
            }
        }
    }

    static class MyReducer extends Reducer<Text, Text, Text, Text> {
        //        Map<String,String> entries = new HashMap<String, String>();
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            context.write(key, new Text(values.iterator().next()));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();
//        configuration.set("mapred.child.java.opts", "-Xmx5");
//        configuration.setBoolean("mapred.output.compress", true);
//        configuration.set("mapred.output.compression.type", SequenceFile.CompressionType.BLOCK.toString());
//        configuration.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);

        if (args.length != 2) {
            logger.error("NEED 2 DIRECTORY ARGUMENTS: -input1(posts/comments) -OUTPUT>");
            System.exit(1);
        }
        Job job = new Job(configuration, "PreprocessText");

        job.setJarByClass(PreprocessText.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);

    }
}
