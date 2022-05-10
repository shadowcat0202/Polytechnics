public class myLinkedQueue<T> {
    private class Node{
        T data;
        Node next;
        Node(T data){
            this.data = data;
            this.next = null;
        }
    }

    private Node front;
    private Node back;

    myLinkedQueue(){
        this.front = null;
        this.back = null;
    }

    public boolean isEmpty(){
        return front == null;
    }

    public void push(T data){
        Node newNode = new Node(data);
        if(isEmpty()){
            front = newNode;
        }
        else{
            back.next = newNode;
        }
        back = newNode;

    }
    public T pop(){
        if(isEmpty())   return null;
        T data = front.data;
        front = front.next;
        if(front == null)   back = null;
        return data;
    }
    public T peek(){
        if(isEmpty())    return null;
        return front.data;
    }
    public void delete(){
        if(isEmpty()){
            System.out.println("Empty");
        }
        else{
            front = front.next;
            if(front == null) back = null;
        }
    }

}
