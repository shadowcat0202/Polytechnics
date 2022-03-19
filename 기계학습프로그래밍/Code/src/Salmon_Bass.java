import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Queue;

public class Salmon_Bass {
    public Salmon_Bass(){
        //생성자
    }

    public void start() throws IOException {
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

            //bass = 0 Salmon = 1
            if(parse[0].equals("Salmon"))     fish_type[i] = true;
            else    fish_type[i] = false;

            //길이 저장
            fish_langth[i] = Integer.parseInt(parse[1]);

            if(min > Integer.parseInt(parse[1]))    min = Integer.parseInt(parse[1]);
            if(max < Integer.parseInt(parse[1]))    max = Integer.parseInt(parse[1]);
        }

        //Salmon과 Bass 히스토그램 분리
        int[] Salmon_histogram = new int[max - min + 1];
        int[] Bass_histogram = new int[max - min + 1];

        //Salmon와 Bass에 대한 히스토그램을 각각 분리
        for(int i = 0; i < fish_langth.length; i++){
            if(fish_type[i])    Salmon_histogram[fish_langth[i] - min]++; //물고기 종류에 맞춰 길이에 맞게 +1씩 증가
            else    Bass_histogram[fish_langth[i] - min]++;  //-min 을 해주는 이유는 histogram을 최대 최소에 맞춰 size fit했기 때문이다
        }

        //무식하게 돌리방법
        /*
        int max_hit = 0;
        int hit = 0;
        for(int i = 2; i <= 26; i++){
            hit = get_hit_use_array(i, fish_langth, fish_type)[0];
            max_hit = (max_hit < hit) ? hit : max_hit;
        }
         */


        System.out.printf("최종 x = %d\n", use_DP(Salmon_histogram, Bass_histogram));


        //draw_histogram(Salmon_histogram, Bass_histogram, min);    //히스토그램 그림 그려주기
    }

    //해당 예측 x값에서의 hit과 miss값을 반환해주는 함수
    public int[] get_hit_use_array(int len_model, int[] X, boolean[] y){
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

    //DP를 사용해서 해보자
    public int use_DP(int[] Salmon_histo, int[] Bass_histo){
        //DP 작업
        //+2를 한 이유는 배열의 양 끝에 0을 넣어서 사용하기 위함
        int[] Salmon_dp = new int[Salmon_histo.length + 2];
        int[] Bass_dp = new int[Bass_histo.length + 2];

        //dp의 각 양 끝값 0으로 초기화
        Salmon_dp[0] = 0;
        Bass_dp[Bass_dp.length - 1] = 0;

        for(int i = 0; i < Salmon_histo.length; i++){
            Salmon_dp[i + 1] = Salmon_histo[i] + Salmon_dp[i];
        }
        for(int i = Salmon_histo.length - 1; i >= 0; i--){
            Bass_dp[i + 1] = Bass_histo[i] + Bass_dp[i + 2];
        }


//        System.out.println("dp\thisto");
//        System.out.println("salmon");
//        for(int i = 0; i < Salmon_histo.length; i++){
//            System.out.printf("%2d\t%3d\t%3d\n", i, Salmon_dp[i+1], Salmon_histo[i]);
//        }
//        System.out.println("Bass");
//        for(int i = 0; i < Bass_histo.length; i++){
//            System.out.printf("%2d\t%3d\t%3d\n", i, Bass_dp[i+1], Bass_histo[i]);
//        }

//        System.out.printf("Salmon dp =\t%s\n", Arrays.toString(Salmon_dp));
//        System.out.printf("Bass dp \t=%s\n", Arrays.toString(Bass_dp));


        int max_hit = 0;    //최대 hit
        int hit = 0;        //맞춘 횟수
        int model_x = 0;    //최종 모델

        //dp는 양 끝에 0이 더 있기 때문에 1부터 25까지 돌아야한다
        for(int i = 1; i <= Salmon_histo.length; i++){
            hit = 0;    //hit의 최대값을 구하기 위해서 for문을 돌때마다 초기화 해주어야 한다
            hit = Salmon_dp[i] + Bass_dp[i+1];  //hit은 결과적으로
            //System.out.printf("x = %d, hit = %d\n", i + 1, hit);  //보고 싶다

            if(max_hit < hit)   {   //최대 hit 갱신
                model_x = i + 1;  //최소값 만큼 밀어주기
                max_hit = hit;
            }
        }

        System.out.println("model_x = " + model_x + " max_hit = " + max_hit);
        return model_x;
    }



    //히스토그램 그려주는 함수
    public void draw_histogram(int[] Salmon_histo, int[] Bass_histo, int min){
        System.out.println("■ = Bass, □ = Salmon");
        for(int i = 0; i < Bass_histo.length; i++){
            if(Bass_histo[i] > 0 ||  Salmon_histo[i] > 0){  //둘중 하나라도 값이 있다면 출력

                //문자열을 한번에 출려하기 위한 StringBuffer => 그냥 System.out.println()을 사용하는것 보다 속도가 빠르다
                StringBuffer sb = new StringBuffer();

                System.out.printf("%2d\t|", i + min);
                for(int d = 0; d < Bass_histo[i]; d++){
                    sb.append("■");
                }
                System.out.println(sb);

                sb = new StringBuffer();    //다시 생성해주어야 한다(출력하더라도 이 작업을 하지 않으면 이어서 나오게 된다)
                System.out.printf("  \t|");
                for(int d = 0; d < Salmon_histo[i]; d++){
                    sb.append("□");
                }
                System.out.println(sb);
            }
        }
    }





}
