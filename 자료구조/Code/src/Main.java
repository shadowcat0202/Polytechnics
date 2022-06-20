import java.util.ArrayList;
import java.util.Arrays;

import java.util.EmptyStackException;
import java.util.Stack;
import Problem.Search;
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
    enum Week {MON, TUE, WED, THU, FRI, SAT, SUN}
    public int[] make_iarr(int arr_size, int max_value){
        int[] arr = new int[arr_size];
        for(int i = 0; i < arr_size; i++){
            arr[i] = (int)(Math.random() * max_value);
        }
        return arr;
    }


    public void quick_sort_test(){
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

    public boolean contains(char[] arr, char find){
        for(char c : arr){
            if(c == find)   return true;
        }
        return false;
    }

    public boolean solution2(String str){
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
    public int solution1_v1(int n, String str){
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
    public int solution1_v2(int n, String str){
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

    public void solution1_v3(){
        String inputString = "((()))";
        int n = 2;
        int ans = 0;
        int cnt = 0;
        Stack<Integer> st = new Stack<>();
        char[] inputchar = inputString.toCharArray();
        try{
            for(int i = 0; i < inputchar.length; i++){
                if(inputchar[i] == '('){
                    st.push(++cnt);
                }
                else{
                    if(st.pop() == n)  ans = i+1;
                }
            }
        }
        catch(EmptyStackException e){
            ans = 0;
        }
        finally {
            if(!st.isEmpty())   ans = 0;
        }
        System.out.println(ans);
    }

    public static int[] arrMinMax(int[] arr){
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        for(int v : arr){
            if(max < v) max = v;
            if(min > v) min = v;
        }
        return new int[] {min, max};
    }

    public static void main(String[] args){
//        int[][] map = {
//                {0, 1, 1, 0, 2, 0, 0},
//                {0, 1, 1, 0, 2, 0, 2},
//                {1, 1, 1, 0, 2, 0, 2},
//                {0, 0, 0, 0, 2, 2, 2},
//                {0, 3, 0, 0, 0, 0, 0},
//                {0, 3, 3, 3, 3, 3, 0},
//                {0, 3, 3, 3, 0, 0, 0}
//        };
        Search sr = new Search();
//        int[] ans = sr.solution_use_recursive(map);
//
//        System.out.printf("영역의 개수: %d\n가장 큰 영역: %d",ans[0], ans[1]);
//        int[] arr = {5,3,76,87,7,5,4,3,32,23,6,6};
//
//        Quick_sort q = new Quick_sort();
//        int[] buf = Arrays.copyOf(arr, arr.length);
//        q.quickSort(buf, 0, arr.length-1, false);
//        System.out.println(Arrays.toString(buf));
//
//        buf = Arrays.copyOf(arr, arr.length);
//        q.quickSort(buf, 0, arr.length-1, true);
//        System.out.println(Arrays.toString(buf));

//        MyTree<String> tree = new MyTree<>();
//        tree.insert("-");
//        tree.insert("*");
//        tree.insert("/");
//        tree.insert("A");
//        tree.insert("B");
//        tree.insert("C");
//        tree.insert("D");
//
//        System.out.println("전위 Order: ");
//        tree.preorder();
//        System.out.println("\n중위 Order: ");
//        tree.inorder();
//        System.out.println("\n후위 Order: ");
//        tree.postorder();

        Week today = Week.valueOf("TUE");
        System.out.println(today);
        int[] arr = {1,2,3,4,5,6};
        int[] result = arrMinMax(arr);
        System.out.println(result[0]+ ","+ result[1]);


    }
}
