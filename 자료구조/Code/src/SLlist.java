class Node<T>{
    T data;
    Node<T> next;

    Node(T data){
        this.data = data;
        this.next = null;
    }
}
public class SLlist<T> {
    private Node<T> head;
    private Node<T> tail;
    private int size;

    public SLlist(){
        this.head = null;
        this.tail = null;
        this.size = 0;
    }
    private Node<T> search(int index){
        //인덱스를 가지고 왔으니 판단을 해야한다
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();

        Node<T> temp = head;
        while(index-- > 0){
            temp = temp.next;
        }
        //같은 코드
//        for(int i = 0; i < index; i++){
//            temp = temp.next;
//        }
        return temp;
    }

    public void addFirst(T value){
        Node<T> newNode = new Node<>(value);
        newNode.next = this.head;
        this.head = newNode;
        this.size++;

        if(this.head.next == null)   tail = head;
    }
    public void addLast(T value){
        if(this.size == 0){
            addFirst(value);
            return;
        }
        Node<T> newNode = new Node<>(value);

        tail.next = newNode;
        tail = newNode;
        size++;
    }

    public boolean add(T value){
        addLast(value);
        return true;
    }
    public void add(int index, T value){
        if(index < 0 || index > size)   throw new IndexOutOfBoundsException();
        if(index == 0)  addFirst(value);
        if(index == size)   addLast(value);

        Node<T> prev_Node = search(index-1);
        Node<T> next_Node= prev_Node.next;
        Node<T> newNode = new Node<>(value);

        prev_Node.next = null;
        prev_Node.next = next_Node;
        newNode.next = next_Node;
        size++;
    }


    public T remove(){
        if(this.head == null)   throw new IndexOutOfBoundsException();

        T element = head.data;
        Node<T> nextNode = head.next;
        head.data = null;
        head.next = null;
        size--;

        if(size == 0)   tail = null;
        return element;
    }
    public T remove(int index){
        if(index == 0)  return remove();
        if(index < 0 || index >= size)  throw new IndexOutOfBoundsException();

        Node<T> prev_Node = search(index -1);
        Node<T> remove_Node = prev_Node.next;
        Node<T> next_Node = remove_Node.next;

        T element = remove_Node.data;

        prev_Node.next = next_Node;

        if(prev_Node.next == null)  tail = prev_Node;

        remove_Node.data = null;
        remove_Node.next = null;
        size--;
        return element;
    }


    public T get(int index){
        return search(index).data;
    }

    public boolean contains(T value){return indexOf(value) >= 0;}

    public int indexOf(T value){
        int index = 0;
        for(Node<T> x = head; x != null; x = x.next, index++){
            if(value.equals(x.data))    return index;
        }
        return -1;
    }

    public void clear(){
        for(Node<T> x = head; x != null;){
            Node<T> nextNode = x.next;
            x.data = null;
            x.next = null;
            x = nextNode;
        }
        head = tail = null;
        size = 0;
    }
    public String toString(){
        StringBuilder sb = new StringBuilder();
        if(this.size > 0){
            sb.append("[");
            for(Node<T> x = head; x != null; x = x.next){
                sb.append(x.data).append(" ");
            }
            sb.delete(sb.length()-1, sb.length());
            sb.append("]");
        }else{
            sb.append("[]");
        }
        return sb.toString();
    }



}
