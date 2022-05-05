import java.util.Arrays;
<<<<<<< HEAD
=======
import java.util.Stack;

>>>>>>> 367930bd7abb1210c0f4aa3318a60d33fb395f8b
class test{
    private int data;
    test(int i){
        this.data = i;
    }
    public int getData(){
        return this.data;
    }
}
public class Main {
    public static int[] make_iarr(int arr_size, int max_value){
        int[] arr = new int[arr_size];
        for(int i = 0; i < arr_size; i++){
            arr[i] = (int)(Math.random() * max_value);
        }
        return arr;
    }


    public static void quick_sort_test(){
        int[] arr;  //오리지날 배열
        int[] buf;  //동일한 측정을 위한 임시 배열
        float[] time_sum = new float[3];    //좌우중간 시간측정
        int Test_case = 1;
        my_sort ms = new my_sort();
        for(int i = 0; i < Test_case; i++){
            arr = make_iarr(10, 100);  //새로운 랜덤 배열 생성
//            System.out.println(Arrays.toString(arr)); //오리지날 배열
            for(int j = 0; j < 3; j++){
                buf = arr;  //동일한 랜덤 배열을 기반으로
                ms.Quick_sort(buf, false, j);   //좌우중간 한번씩 돌아가면서 정렬
//                System.out.print(j+":");  //어떤 피벗
//                System.out.println(Arrays.toString(buf) );    //정렬 후 배열
//                System.out.println(); 
                time_sum[j] += (float)ms.getRuntime()/1000; //각 정렬마다 시간 측정값 더하기
//                System.out.println(ms.isSort(arr));   //정렬 여부
            }
//            System.out.println("\n");

        }
        System.out.println("왼쪽\t\t\t오른쪽\t\t중간\t");
        for(float value : time_sum) {   //속도 평균 구하기
            System.out.printf("%f(s)\t", value / (float) Test_case);
        }
        System.out.println();

    }
<<<<<<< HEAD


    public static void main(String[] args){
        SLlist<test> ls = new SLlist<>();
        for(int i = 0; i < 5; i++){
            ls.add(new test(i));
        }

        for(int i = 0; i < 5; i++){
            System.out.println(ls.get(i).getData());
        }
        ls.
=======
    public static boolean contains(char[] arr, char find){
        for(char c : arr){
            if(c == find)   return true;
        }
        return false;
    }

    public static boolean solution2(String str){
        Stack<Character> st = new Stack<>();
        char[] open = {'[', '{', '('};
        char[] close = {']', '}', ')'};

        for(int i = 0; i < str.length(); i++){
            if(contains(open, str.charAt(i))){
                st.push(str.charAt(i));
            }
            else if(contains(close, str.charAt(i))){
                if(st.peek() != str.charAt(i))  return false;
            }
        }
        if(!st.isEmpty())   return false;

        return true;
    }
    //괄호문제 초안?
    public static int solution1_v1(int n, String str){
        class pair{
            char c;
            int num;
            pair(char c, int num){
                this.c = c;
                this.num = num;
            }
        }
        Stack<pair> st = new Stack<pair>();
        int open_count = 0;
        int res = 0;
        for(int i = 0; i < str.length();i++){
            if(str.charAt(i) == '('){
                st.push(new pair('(', ++open_count));
            }else if(str.charAt(i) == ')'){
                if(st.isEmpty())    return 0;
                if(st.pop().num == n)  res = i;
            }
        }
        if(!st.isEmpty())   return 0;

        return res+1;
    }
    //괄호 문제 개선안
    public static int solution1_v2(int n, String str){
        int open_cnt = 0;
        int res = 0;
        Stack<Integer> st = new Stack<>();
        for(int i= 0; i < str.length(); i++){
            if(str.charAt(i) == '('){
                st.push(++open_cnt);
            }
            else{
                if(st.isEmpty())    return 0;
                if(st.pop() == n)  res = i;
            }
        }
        if(!st.isEmpty())   return 0;

        return res+1;
    }

    public static void main(String[] args){
        int[] input_int = {3,2,6,4};
        String[] input_str = {
                "()((()()))",
                "))()(())",
                "((())((()))(()))",
                "(((((((())))))))"
        };
        for(int i = 0; i < input_str.length; i++){
            System.out.println(solution1_v2(input_int[i], input_str[i]));
        }
//        System.out.println(solution1_v2(input_int[0], input_str[0]));


//        String input = "{(A+B)-3}*5 + [{cos(x+y)+7}-1]*4";
//
//        System.out.println(solution2(input));
>>>>>>> 367930bd7abb1210c0f4aa3318a60d33fb395f8b


    }
}
