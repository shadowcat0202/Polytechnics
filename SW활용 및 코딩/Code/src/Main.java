import java.util.Scanner;

public class Main {

    public static int sum(int i, int j){

        System.out.println("변경전 =" + System.identityHashCode(i));
        System.out.println("함수 내에서의 i =" + ++i);
        System.out.println("변경전 ="+ System.identityHashCode(i));
        return i + j;
    }
    public static void stub(){
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
        stub();
    }
}
