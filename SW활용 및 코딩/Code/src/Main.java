public class Main {

    public static int sum(int i, int j){

        System.out.println("변경전 =" + System.identityHashCode(i));
        System.out.println("함수 내에서의 i =" + ++i);
        System.out.println("변경전 ="+ System.identityHashCode(i));
        return i + j;
    }

    public static void main(String[] args){
        int i, j;
        char a;
        String b;
        final int TEN = 10;
        i = 1;
        System.out.println(System.identityHashCode(i));
        j = sum(i, TEN);
        System.out.println("함수 밖에서 i = " + i);
        a = '?';
        b = "Hello";
        System.out.println(System.identityHashCode(i));

        System.out.println(b);
        System.out.println(TEN);
        System.out.println(j);
    }
}
