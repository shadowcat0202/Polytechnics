import java.util.Scanner;

public class Main {
    public void dowhile(){
        char c = 'a';
        do{
            System.out.print(c++ + ",");
            }while(c <= 'z');
        System.out.println();
    }
    public void arr2d(){
        int[][] a2d = new int[2][]; //초기 공간이 없어서 해줘야 주소값이 나온다
        a2d[0] = new int[3];
        a2d[1] = new int[4];
        System.out.println(a2d);
        System.out.println(a2d[0] + ", " + a2d[1]);
        int[][] arr = new int[2][];
        arr[0] = new int[3];
        arr[1] = new int[2];
        arr = a2d;
        System.out.println(arr);


    }

    public int sum(int i, int j){

        System.out.println("변경전 =" + System.identityHashCode(i));
        System.out.println("함수 내에서의 i =" + ++i);
        System.out.println("변경전 ="+ System.identityHashCode(i));
        return i + j;
    }

    public void stub(){
        char grade;
        int score;
        Scanner sc = new Scanner(System.in);
        System.out.println("반복을 끝내려면 0~100을 제외한 숫자를 입력하세요:");
        while(true){
            System.out.print("점수:");
            score = sc.nextInt();
            if(score > 100 || score < 0)    break;

            if(score >= 90) grade = 'A';
            else if(score >= 80) grade = 'B';
            else if(score >= 70) grade = 'C';
            else if(score >= 60) grade = 'D';
            else grade = 'F';

            System.out.println("성적은" + grade);
        }
    }

    public static void main(String[] args){
        Main m = new Main();
        m.arr2d();
    }
}
