import java.io.IOException;
import java.util.Random;

public class Main {
    public static int randNum(int min , int max){


        //Random 객체의 인스턴스 rand
        Random rand = new Random();
        int num = rand.nextInt();
        //최대값 - 최소값 + 최소값
        //랜덤 구하는 공식
        num = (num >>> 1) % (max - min) + min;
        //int 형으로 값을 num에 반환한다.
        return num;
    }

    public static void main(String[] args) throws IOException {
        Salmon_Bass ts = new Salmon_Bass();
        ts.start();

        //stub();
//        bass_Salmon bs = new bass_Salmon("./salmon_bass_data.csv");
//        bs.read_data_draw_histogram();

        //Salmon 2 ~ 15
        //Bass 7 ~ 24
        /*
        int min = 2;
        int max = 15;

        StringBuffer sb = new StringBuffer();
        for(int i = 0; i < 1000; i++){
            sb = new StringBuffer();
            sb.append("Bass,");
            sb.append(Integer.toString(randNum(7, 24)));
            sb.append(",0");
            System.out.println(sb);
        }

         */


    }
}
