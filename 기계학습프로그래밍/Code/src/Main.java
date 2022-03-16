import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void stub() throws IOException {
        int row_number = 318;

        FileReader fr = new FileReader("./salmon_bass_data.csv");
        BufferedReader br = new BufferedReader(fr);


        boolean[] fish_class = new boolean[row_number];
        int[] fish_langth = new int[row_number];

        String line = br.readLine();
        int cnt = 0;
        int[] minmax = {Integer.MAX_VALUE, Integer.MIN_VALUE};

        while((line = br.readLine()) != null){
            String[] parse = line.split(",");


            //bass면 1 아니면 0 (bass 가 길다)
            if(parse[0].equals("bass"))     fish_class[cnt] = true;
            else    fish_class[cnt] = false;

            fish_langth[cnt++] = Integer.parseInt(parse[1]);

            if(minmax[0] > Integer.parseInt(parse[1]))  minmax[0] = Integer.parseInt(parse[1]);
            if(minmax[1] < Integer.parseInt(parse[1]))  minmax[1] = Integer.parseInt(parse[1]);
        }

        float model_length = 10.0f;
        System.out.println("시작 x = " + model_length);
        float rl = 0.1f;
        float E = 1.0f;


        //학습 횟수 최적의 x를 찾을때까지
        for(int epoc = 0; epoc < 10; epoc++) {
            int hit = 0, miss = 0;
            int bf_hit = hit;

            for(int i = 0; i < row_number; i++){
                boolean type;
                
                //예측
                if(fish_langth[i] > model_length){
                    type = true;    //bass
                }else{
                    type = false;   //salmon
                }
                
                //검증? 확인
                if(type == fish_class[i])  hit++;
                else    miss++;

            }
            System.out.printf("hit = %d miss = %d \n", hit, miss);

            //에러율?

            //learing rate
            if(hit > bf_hit) model_length += rl;
            else    model_length -= rl;
        }
        System.out.println("최종 x = " + model_length);





    }

    public static void main(String[] args) throws IOException {
        stub();




    }
}
