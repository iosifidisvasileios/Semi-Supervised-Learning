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
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;

public final class TweetBasicStatistics {

    static Logger logger = Logger.getLogger(TweetBasicStatistics.class);
    private static class MyMapper extends Mapper<LongWritable, Text, Text, Text> {
        protected void map(LongWritable key, Text row, Context context) throws IOException, InterruptedException {
            JSONParser parser = new JSONParser();
            final String items = row.toString();
            context.write(new Text("Total_Rows"), new Text("1"));

            try {
                final JSONObject  obj = (JSONObject) parser.parse(items);
                if (obj.containsKey("delete")) {
                    context.write(new Text("Deleted"), new Text("1"));
                }
                else{
                    try {
                        if (obj.get("lang").equals("en")) {
                            String flag;
                            if (obj.containsKey("retweeted_status")) {
                                flag = "1";
                            } else {
                                flag = "0";
                            }
                            String hashTags = "";

                            try {
                                JSONArray msg = (JSONArray) ((JSONObject) obj.get("entities")).get("hashtags");
                                for (int i = 0; i < msg.size(); i++) {
                                    JSONObject temp = (JSONObject) msg.get(i);
                                    hashTags += temp.get("text") + "-";
                                }
                            }catch(NullPointerException e){
                                logger.error(e);
                            }

                            if (hashTags.equals("")) {
                                hashTags = "null";
                            } else {
                                hashTags = hashTags.substring(0, hashTags.length() - 1);
                            }

                            String text = obj.get("text").toString().replaceAll(",", " ");
                            String concat = obj.get("created_at") + "," + obj.get("geo") + "," + hashTags + "," + text + "," + flag;

                            context.write(new Text("en"), new Text("1"));

                        }else{
                            context.write(new Text(obj.get("lang").toString()), new Text("1"));
                        }

                    }catch (NullPointerException e) {
                        logger.error(e);
                        context.write(new Text("NullPointerException"),new Text("1"));
                    }
                }
            } catch (ParseException e) {
                context.write(new Text("Unable_to_parse"), new Text("1"));
                logger.error(e);

            }
        }
    }

    static class MyReducer extends Reducer<Text, Text, Text, IntWritable> {
        IntWritable result = new IntWritable();
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int sum = 0;

            for (Text val : values) {
                sum += 1;
            }
            result.set(sum);

            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            logger.error("NEED 2 DIRECTORY ARGUMENTS: -input1(posts/comments) -OUTPUT>");
            System.exit(1);
        }
        Configuration configuration = new Configuration();
//        configuration.set("mapred.child.java.opts", "-Xmx4G");
//        configuration.set("yarn.nodemanager.resource.memory-mb", "58000");
//        configuration.set("yarn.scheduler.minimum-allocation-mb", "2000");
//        configuration.set("yarn.scheduler.maximum-allocation-mb", "4000");
//        configuration.set("mapreduce.map.memory.mb", "4096");
//        configuration.set("mapreduce.reduce.memory.mb", "4096");

        Job job = new Job(configuration, "TweetBasicStatistics");

        job.setJarByClass(TweetBasicStatistics.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setMapOutputValueClass(Text.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);


        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);

    }
}
