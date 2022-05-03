public class Test {
    void test(){
        System.out.println("생성자 실행");
    }

    public void start(){
        final double PI = 3.14D;
        System.out.print("hello\n");
        System.out.println("hello\b\n");

        final double IP = 3.14D;

        double radius = 10.0;
        double circleArea = Math.pow(radius, 2) * PI;

        System.out.println(circleArea);

        byte b = 127;
        int i = 100;

        System.out.println(b + i);
        System.out.println(10/4);
        System.out.println(10.0/4);
        System.out.println((char)0x12340041);   //강제 타입 변환 결과 0x41이 되며, char A 코드임
        System.out.println((byte)(b+i));
        System.out.println((int)2.9+1.8);
        System.out.println((int)(2.9 + 1.8));
        System.out.println((int)2.9 + (int)1.8);


    }
}
