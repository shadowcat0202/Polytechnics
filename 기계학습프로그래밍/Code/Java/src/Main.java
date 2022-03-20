import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class Main {
    static final long jo = 1000000000000L;

    public static void person_mid() throws IOException {
        FileReader fr = new FileReader("./2022인구추계(중위).csv");
        BufferedReader br = new BufferedReader(fr);

        int person[] = new int[101];
        String line = br.readLine();
        line = br.readLine();

        int idx = 0;
        while((line = br.readLine())!= null){
            String[] parse = line.split(",");
            person[idx++] = Integer.parseInt(parse[1]) + Integer.parseInt(parse[2]);
        }

        int l = -1, r = 101;
        int l_sum = 0, r_sum = 0;

        while(l < r){
            //System.out.println(l_sum + "," + r_sum + " == "+(l-1) + ", " + (r+1));
            if(l_sum < r_sum)   {
                l_sum += person[++l];
            }
            else if(l_sum > r_sum){
                r_sum += person[--r];
            }
            else{
                l_sum += person[++l];
                r_sum += person[--r];
//                if(l > r){
//                    l--;
//                    break;
//                }
            }
        }
        System.out.println("중위 연령대는 = " + (100-l));



    }

    public static void stub(){
        int samsong = 50000;
        long total = 298 * jo;
        int ju = 10;
        int sum = samsong * ju;
        System.out.println(total);

        String bass = "180";
        String Salmon = "140";

        System.out.println("문자열 그대로 더할거임:" +bass + Salmon);

        System.out.println("숫자로 더할거임: "+ (Integer.parseInt(bass) + Integer.parseInt(Salmon)));

        String s = "Hello";
        String j = "Java";

        System.out.println(s + "!" + j);
        System.out.printf("%s! %s\n",s,j);

        int money = 48584;
        int woal = 36;
        System.out.println(money + woal);

    }


    public static void main(String[] args) throws IOException {
//        Salmon_Bass ts = new Salmon_Bass();
//        ts.start();
        person_mid();
    }


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
}
