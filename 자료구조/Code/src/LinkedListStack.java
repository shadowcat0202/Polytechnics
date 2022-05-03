import java.util.EmptyStackException;

public class LinkedListStack<T> {
    private class Node{
        T data;
        Node next;

        Node(T data){
            this.data = data;
            this.next = null;
        }
    }

    private Node top;

    LinkedListStack(){
        top = null;
    }

    private void rmNode(){
        Node rm = top;
        top = top.next;
        rm.data = null;
        rm.next = null;
        rm = null;
    }

    boolean isEmpty(){  return top == null; }

    void push(T value){
        Node psNode = new Node(value);
        psNode.next = top;
        top = psNode;
    }

    T pop(){
        if(isEmpty())   throw new EmptyStackException();

        T data = top.data;
        rmNode();
        return data;
    }

    T peek(){
        if(isEmpty())   return null;
        return top.data;
    }

    void del(){
        if(isEmpty())   throw new EmptyStackException();
        rmNode();
    }
}
