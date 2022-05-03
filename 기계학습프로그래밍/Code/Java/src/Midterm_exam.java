import java.util.HashMap;
import java.util.Iterator;
import java.util.Arrays;

class Answer{
    int [] solution1(String[] shirt_size){
        int len = shirt_size.length;
        System.out.println("입력 배열 길이는" + len + "입니다");
        int[] result = new int[6];
        String[] sizeName = {"XS", "S", "M", "L", "XL", "XXL"};
        for(String size : shirt_size){
            switch (size) {
                case "XS" -> result[0]++;
                case "S" -> result[1]++;
                case "M" -> result[2]++;
                case "L" -> result[3]++;
                case "XL" -> result[4]++;
                case "XXL" -> result[5]++;
            }
        }

//        enum mysize {XS, S, M, L, XL, XXL}
//        for(int i = 0; i < shirt_size.length; i++){
//            mysize tmp = mysize.valueOf(shirt_size[i]);
//            result[tmp.ordinal()]++;
//        }

        return result;
    }

    int [] solution2(int[] original){
        int len = original.length;
        System.out.println("입력 배열 길이는" + len + "입니다");

        int[] result = new int[len];
        for(int i = 0; i < len; i++){
            result[i] = original[len-i-1];
        }
        return result;
    }

    int solution3(int n){
        System.out.println(n+"항 까지의 합을 구합니다.");
        int result = 0;
        for(int i = 0; i < n; i++){
            result += (i * 4) + 2;
        }

        return result;
    }

    int solution4(int[] original){
        int len = original.length;
        System.out.println("입력 배열 길이는" + len + "입니다");
        if(len == 0)    return 0;

        int result = 0;
        int max = 1;
        int min = Integer.MAX_VALUE;

        // use HashMap=======================================
//        HashMap<Integer, Integer> map = new HashMap<>();
//        for(int num: original){
//            if(!map.containsKey(num))   map.put(num, 1);
//            else    map.replace(num, map.get(num)+1);
//        }
//        Iterator<Integer> keys = map.keySet().iterator();
//        while(keys.hasNext()){
//            int key = keys.next();
//            int value = map.get(key);
//            if(value < min) min = value;
//            if(value > max) max = value;
//        }

        // non use HashMap 1=======================================
        //숫자 크기가 유한하다고 가정할때 사용하는 방법 => 비효율
//        int[] count = new int[1000+1];
//        for(int num : original){
//            count[num]++;
//        }
//
//        for(int i = 1; i < count.length; i++){
//            if(count[i] > 0 && count[i] < min)  min = count[i];
//            if(count[i] > max)  max = count[i];
//        }

        //non use hashMap 2=======================================
        int[] tmp = new int[1]; // 숫자가 뭔가 하나는 들어올 것이다
        Arrays.sort(original);  // n log n
        int remember = original[0];
        tmp[0] = 1; //for문을 1부터 시작하기 때문에 미리 1을 넣어놓고 시작해야한다
        for(int i = 1; i < len; i++){
            if(original[i] != remember) {
                tmp = Arrays.copyOf(tmp, tmp.length + 1);
                remember = original[i];
            }
            tmp[tmp.length-1]++;
        }

        for (int value : tmp) {
            if (value < min) min = value;
            if (value > max) max = value;
        }
        result = max / min;
        return result;
    }
}
public class Midterm_exam{
    public static void main(String[] args){
        System.out.println("AI-Engineering - 전세환");

        Answer ans = new Answer();

        /*-----------------문제 1번 테스트-------------------*/
        String[] param1 = {"XS", "S", "L","L", "XL", "XXL"};
        int[] result1 = ans.solution1(param1);

        System.out.print("[");
        for(int i = 0; i < result1.length; i++){
            System.out.print(result1[i] + ", ");
        }
        System.out.println("]");

        /*-----------------문제 2번 테스트-------------------*/
        int[] param2 = {4,5,2,6,7};
        int[] result2 = ans.solution2(param2);

        System.out.print("[");
        for(int i = 0; i < result2.length; i++){
            System.out.print(result2[i] + ", ");
        }
        System.out.println("]");

        /*-----------------문제 3번 테스트-------------------*/
        int param3 = 1;
        int result3 = ans.solution3(param3);
        System.out.println(param3 + "항 까지의 합은" + result3 + "입니다");

        /*-----------------문제 4번 테스트-------------------*/
//        int[] param4 = {1,2,3,3,1,3,3,2,3,2};
        int[] param4 = {1,2,1,2,1,20000,20000,1};
        int result4 = ans.solution4(param4);

        System.out.println("가장 적게 나온 원소와 가장 큰 원소는 " + result4 + "배 차이 입니다.");

    }
}
