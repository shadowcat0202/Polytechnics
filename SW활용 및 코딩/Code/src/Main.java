import java.util.Scanner;
import package1.*;
import package2.*;

class pizza{
    private int r = 0;
    private String name = "";

    pizza(String name, int r){
        this.name = name;
        this.r = r;
    }
    private double get_round(){
        return 3.14 * 2 * r;
    }
    public void show(){
        StringBuilder sb = new StringBuilder();
        sb.append(this.name).append("의 둘레는 ").append(String.format("%.2f", get_round())).append("cm");
        System.out.println(sb);
    }
}
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
    public void show(){
        System.out.println("wow");
    }
    public static void show_1(){
        System.out.println("wowo2");
    }

    public int sum(int i, int j){

        System.out.println("변경전 =" + System.identityHashCode(i));
        System.out.println("함수 내에서의 i =" + ++i);
        System.out.println("변경전 ="+ System.identityHashCode(i));
        return i + j;
    }

    public void stub(){
        int[][] arr = new int[2][];
        arr[0] = new int[] {1,2,3};
        arr[1] = new int[] {4,5,6,7};
        for(int[] arr1d : arr){
            for(int ele : arr1d){
                System.out.print(ele + " ");
            }
            System.out.println();
        }


    }
    public static void main(String[] args){
        pizza g = new pizza("맛있는 피자", 10);
        pizza b = new pizza("맛없는 피자", 2);
        g.show();
        b.show();

    }

}
