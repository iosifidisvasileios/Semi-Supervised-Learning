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

public final class StopwordCounter {

    private static final ProcessingOfRow processingOfRow = new ProcessingOfRow();
    static Logger logger = Logger.getLogger(StopwordCounter.class);

    private static class MyMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        protected void map(LongWritable key, Text row, Context context) throws IOException, InterruptedException {

            final String items = row.toString();

            try {
                String preprocessed = "";
                String outpout ="";

                String[] stringArray = items.split(",");

                final String idKey = String.valueOf(Long.valueOf(stringArray[0].split("\t")[0]));

                if (!stringArray[2].equals("null")){
                    preprocessed = stringArray[6];
                    outpout = "," + stringArray[1] + "," + stringArray[2] + "," + stringArray[3]+ "," + stringArray[4]+ "," + stringArray[5] + "," + stringArray[6] + "," + stringArray[7];
                }else {
                    preprocessed = stringArray[4];
                    outpout = "," + stringArray[1] + "," + stringArray[2] + "," + stringArray[3]+ "," + stringArray[4]+ "," + stringArray[5];
                }

                //remove [!?.] for better word parsing

                //remove #, @, links
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


//                preprocessed = processingOfRow.removeStopwords(preprocessed.split(" "));
//
//                String emotion ="";
//
//                if (happyResult.getCounter() == 0 && sadResult.getCounter() == 0){
//                    emotion = "neutral";
//                }else if(happyResult.getCounter() > 0 && sadResult.getCounter() == 0){
//                    emotion = "positive";
//                }else if(sadResult.getCounter() > 0 && happyResult.getCounter() == 0){
//                    emotion = "negative";
//                }else if(happyResult.getCounter() > 0 && happyResult.getCounter() > sadResult.getCounter()){
//                    emotion = "semi_positive";
//                }else if(sadResult.getCounter() > 0 && sadResult.getCounter()> happyResult.getCounter()){
//                    emotion = "semi_negative";
//                }else{
//                    emotion = "unknown";
//                }
//
//                final String emoticonsList = happyResult.getEmoticons() + sadResult.getEmoticons();
//                outpout += "," + emotion + "," + emoticonsList + "," + preprocessed;
//
                for(String word : preprocessed.split(" ")) {
                    context.write(new Text(word), new IntWritable(1));
                }

            }catch(NumberFormatException e){
                logger.info(e);
//                context.write(new Text("NumberFormatException"),new Text("1"));
//                cnt += 1;
            }catch (ArrayIndexOutOfBoundsException e){
                logger.info(e);
//                context.write(new Text("ArrayIndexOutOfBoundsException"),new Text("1"));
//                cnt2 += 1;
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "stopwordsLoader");

        job.setJarByClass(StopwordCounter.class);
        job.setMapperClass(MyMapper.class);

        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
