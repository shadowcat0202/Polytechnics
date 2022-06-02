package Interface;

public interface PhoneInterface {
    final int TIMEOUT = 10000;  // 인터페이스의 변수는 사용못한다 ==> 상수만 사용가능하다 final 생략 가능
    public void sendCall();     // 인터페이스 내의 메서드 접근 지정자 생략 가능
    void receiveCall();
    default void print_Logo(){  // 추상메서드에서 정의할 메서드는 default 필요 물론 Override 가능
        System.out.println("** Phone **");
    }
}


