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
        Main m = new Main();
        m.stub();
    }
}
