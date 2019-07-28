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
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.EOFException;
import java.io.IOException;

import static org.apache.commons.lang3.StringUtils.normalizeSpace;

public final class ExtractEnglishTweets {

    static Logger logger = Logger.getLogger(ExtractEnglishTweets.class);

    private static class MyMapper extends Mapper<LongWritable, Text, Text, Text> {

        @Override
        public void run(Context context) {
            try {
                setup(context);
                while (context.nextKeyValue()) {
                    map(context.getCurrentKey(), context.getCurrentValue(), context);
                }
                cleanup(context);
            } catch (Exception e) {
                logger.warn(e,e);
            }
        }



        protected void map(LongWritable key, Text row, Context context) throws IOException, InterruptedException {
            try {

                JSONParser parser = new JSONParser();
                final String items = row.toString();

                JSONObject  obj = (JSONObject) parser.parse(items);
                if (!obj.containsKey("delete")){
                    try {
                        if (obj.get("lang").equals("en")) {

                            boolean secondLang = true;
                            if (obj.containsKey("retweeted_status")) {
                                obj = (JSONObject) obj.get("retweeted_status");

                                if (!obj.get("lang").equals("en")) {
                                    secondLang = false;
                                }
                            }

                            if(secondLang) {
/*                                String hashTags = "";
                                try {
                                    JSONArray msg = (JSONArray) ((JSONObject) obj.get("entities")).get("hashtags");
                                    for (int i = 0; i < msg.size(); i++) {
                                        JSONObject temp = (JSONObject) msg.get(i);
                                        hashTags += temp.get("text") + "-";
                                    }
                                } catch (NullPointerException e) {

                                }

                                if (hashTags.equals("")) {
                                    hashTags = "null";
                                } else {
                                    hashTags = hashTags.substring(0, hashTags.length() - 1);
                                }
*/

                                JSONObject user = (JSONObject) obj.get("user");
                                String tweetId = obj.get("id_str").toString();
                                String username = user.get("screen_name").toString();
                                String date = obj.get("created_at").toString();
                                int followers_count = Integer.valueOf(user.get("followers_count").toString());
                                int friends_count = Integer.valueOf(user.get("friends_count").toString());

                                int retweet_count = Integer.valueOf(obj.get("retweet_count").toString());
                                int favorite_count = Integer.valueOf(obj.get("favorite_count").toString());
                                String text = normalizeSpace(obj.get("text").toString().replaceAll("\\t", " ").replaceAll("\\r", " ").replaceAll("\\n", " ")).replaceAll("  ", " ");

                                String concat = username + "\t" + date + "\t" + followers_count + "\t" + friends_count + "\t" + retweet_count + "\t" + favorite_count + "\t" + text;
                                context.write(new Text(tweetId), new Text(concat));
                            }
                        }
                    }catch (NullPointerException e) {}}}
            catch (ParseException e) {}
            catch (EOFException e){}
        }
    }
    static class MyReducer extends Reducer<Text, Text, Text, Text> {
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            context.write(key, new Text(values.iterator().next()));
        }
    }
    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();
        configuration.set("mapreduce.output.basename", "summer");
        configuration.set("mapreduce.max.map.failures.percent", "50");
        configuration.set("mapreduce.map.failures.maxpercent", "50");

        Job job = new Job(configuration, "ExtractEnglishTweets");
        job.setJarByClass(ExtractEnglishTweets.class);

        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
