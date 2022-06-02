import java.util.Scanner;

import Interface.MP3Interface;
import Interface.MobliePhoneInterface;
import package1.*;
import package2.*;
import java.util.Arrays;
import Interface.PhoneInterface;

class PDA{
    public int calc(int x, int y){
        return x + y;
    }
}
class smartPhone extends PDA implements MobliePhoneInterface, MP3Interface{

    @Override
    public void play() {
        System.out.println("음악 연주");
    }

    @Override
    public void stop() {
        System.out.println("음악 중단");
    }

    @Override
    public void sendSMS() {
        System.out.println("문자 보냄");
    }

    @Override
    public void receiveSMS() {
        System.out.println("문자 받음");
    }

    @Override
    public void sendCall() {
        System.out.println("전화 건다");
    }

    @Override
    public void receiveCall() {
        System.out.println("전화 받는다");
    }
    public void schedule(){
        System.out.println("일정 관리");
    }
}

class Point{
    private int x, y;
    public Point(){
        this.x = this.y = 0;
    }
    public Point(int x, int y){
        this.x = x;
        this.y = y;
    }

    public void showPoint(){
        System.out.println("pos:(" + this.x + ", " + this.y + ")");
    }
}

class ColorPoint extends Point{
    private int R, G, B;
    public ColorPoint(int x, int y, int r, int g, int b){
        super(x,y);
        try{
            if(r < 0 || g < 0 || b < 0 || r > 255 || g > 255 || b > 255)
                throw new Exception("color int: 0 to 255");
        }
        catch(Exception e){
            System.out.println(e);
        }
        this.R = r;
        this.G = g;
        this.B = b;
    }
    public int[] get_color_array(){
        return new int[] {this.R, this.G, this.B};
    }
    public void showColorPoint(){
        StringBuilder sb = new StringBuilder();
        sb.append("color:(");
        sb.append(this.R).append(", ");
        sb.append(this.G).append(", ");
        sb.append(this.B).append(")");
        showPoint();
        System.out.println(sb.toString());
    }
}

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

class AA{
    AA(){
        System.out.println("Class AA non");
    }
    AA(int a){
        System.out.println("Class AA" + a);
    }
}
class BB extends AA{
    BB(){
        System.out.println("Class BB");
    }
}
class Shape{
    public Shape next;
    public Shape(){
        this.next = null;
    }
    public void draw(){
        System.out.println("Shape");
    }
}
class Line extends Shape{
    public void draw(){
        super.draw();
        System.out.println("Line");
    }
}
class Rect extends Shape{
    public void draw(){
        System.out.println("Rect");
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
        smartPhone phone = new smartPhone();
        phone.play();
        phone.receiveCall();
        System.out.println(phone.calc(3,6));
        phone.schedule();


    }

}
