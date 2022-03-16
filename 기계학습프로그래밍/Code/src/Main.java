import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Queue;

public class Main {

    public static void draw_histogram(int[] histogram, int resize){
        for(int i = histogram.length - 1; i >= 0; --i){
            System.out.printf("%2d\t",i + resize);

            StringBuffer sb = new StringBuffer();
            for(int dot = 0; dot < histogram[i]; dot++){
                if(dot % 5 == 0)    sb.append("|");
                sb.append("■");

                if(dot == histogram[i]-1) sb.append(" " + histogram[i]);

            }
            System.out.println(sb);
        }
    }

    public static int[] get_hit_use_array(int len_model, int[] X, boolean[] y){
        int hit = 0, miss = 0;
        for(int i = 0; i < X.length; i++){
            boolean prediction; //예측값

            //예측
            if(X[i] > len_model)  prediction = true;
            else    prediction = false;

            //검증
            if(prediction == y[i])  hit++;
            else    miss++;
        }
        return new int[]{hit, miss};
    }

    

    public static void stub() throws IOException {
        int row_number = 318;

        FileReader fr = new FileReader("./salmon_bass_data.csv");
        BufferedReader br = new BufferedReader(fr);


        boolean[] fish_type = new boolean[row_number];
        int[] fish_langth = new int[row_number];

        String line = br.readLine();
        int cnt = 0;
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;

        //물고기 길이 저장 + 최소 최대 구하기
        for(int i = 0; (line = br.readLine()) != null; i++){
            String[] parse = line.split(",");

            //bass = 1 Salmon = 0
            if(parse[0].equals("Bass"))     fish_type[i] = true;
            else    fish_type[i] = false;

            //길이 저장
            fish_langth[i] = Integer.parseInt(parse[1]);


            if(min > Integer.parseInt(parse[1]))    min = Integer.parseInt(parse[1]);
            if(max < Integer.parseInt(parse[1]))    max = Integer.parseInt(parse[1]);


        }

        int[] histogram_langth = new int[max - min + 1];
        for(int i = 0; i < row_number; i++){
            histogram_langth[fish_langth[i] - min]++;
        }

        //draw_histogram(histogram_langth, min);    //히스토그램 그림 그려주기


        //System.out.printf("min = %d, max = %d\n", min, max);

        //예측 + 확인
        int len_model = 2;//(min + max) / 2;

        int learning_count = 25;
        int learning_rate = 1;
        int bf_hit = 0;
        Queue<Float> hit_rate;
        float best_hit_rate = 0;

        System.out.println("학습\tmd\thit\tmiss\tBHR");

        for(int epoc = 0; epoc < learning_count; epoc++){   //학습 횟수만큼 돌린다
            int[] hm = get_hit_use_array(len_model, fish_langth, fish_type);    //배열 데이터를 직접 사용해 hit, miss 받아오기
            if((float)hm[0] / row_number > best_hit_rate){
                best_hit_rate = (float)hm[0] / row_number;  //최고 적중률 갱신
            }


            System.out.printf("%3d\t%2d\t%3d\t%3d\t\t%.4f\n", epoc, len_model, hm[0], hm[1], best_hit_rate);

            if(hm[0] > bf_hit)    learning_rate *= 1;
            else if(hm[0] == bf_hit);
            else    learning_rate *= -1;

            len_model += 1;

            bf_hit = hm[0];





        }
    }

    public static void main(String[] args) throws IOException {
        stub();
//        bass_Salmon bs = new bass_Salmon("./salmon_bass_data.csv");
//        bs.read_data_draw_histogram();




    }
}
