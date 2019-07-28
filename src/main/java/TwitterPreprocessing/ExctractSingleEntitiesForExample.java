package TwitterPreprocessing; /**
 * Created by iosifidis on 01.08.16.
 */

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.HashSet;

public final class ExctractSingleEntitiesForExample {
    static Logger logger = Logger.getLogger(ExctractSingleEntitiesForExample.class);
    public static final HashSet<String> famous = new FamousWords().getFamousArray();

    private static class MyMapper extends Mapper<LongWritable, Text, Text, Text> {
        protected void map(LongWritable key, Text row, Context context) throws IOException, InterruptedException {

            final String items = row.toString();
            try {

                String[] stringArray = items.split(",");

                final String idKey = String.valueOf(Long.valueOf(stringArray[0].split("\t")[0]));

                for (String word : stringArray[3].split(" ")){
                    if (famous.contains(word.toLowerCase())){
                        final String output = "," + stringArray[1] + "," +
                                stringArray[3] + "," +
                                stringArray[7] + "," +
                                stringArray[stringArray.length - 1 ];

                        context.write(new Text(idKey), new Text(output));

                    }
                }

            } catch (NumberFormatException e) {
                logger.error(e);
                logger.error(items);
            } catch (ArrayIndexOutOfBoundsException e) {
                logger.error(e);
            }
        }
    }

    static class MyReducer extends Reducer<Text, Text, Text, Text> {
        protected void reduce(Text key, Text values, Context context) throws IOException, InterruptedException {
            context.write(key, new Text(values));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();

        if (args.length != 2) {
            logger.error("NEED 2 DIRECTORY ARGUMENTS: -input1(posts/comments) -OUTPUT>");
            System.exit(1);
        }
        Job job = new Job(configuration, "ExctractSingleEntitiesForExample");

        job.setJarByClass(ExctractSingleEntitiesForExample.class);
        job.setMapperClass(ExctractSingleEntitiesForExample.MyMapper.class);
        job.setReducerClass(ExctractSingleEntitiesForExample.MyReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}