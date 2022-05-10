import java.util.Arrays;
import java.util.NoSuchElementException;

public class myQueue<T> {
    private static final int DEFAULT_CAPACITY = 10;
    private Object[] array;
    private int size;

    private int front;
    private int back;

    public myQueue(){
        this.array = new Object[DEFAULT_CAPACITY];
        this.size = 0;
        this.front = 0;
        this.back = 0;
    }

    private void resize(int newCapacity){
        int arrayCapacity = array.length;

        Object[] newArray = new Object[newCapacity];

        for(int i = 1, j = front + 1; i <= this.size; i++, j++){
            newArray[i] = array[j % arrayCapacity];
        }
        this.array = null;
        this.array = newArray;

        front = 0;
        back = this.size;
    }

    public boolean offer(T item){
        if((back + 1) % array.length == front){
            resize(array.length * 2);
        }
        back = (back + 1) % array.length;
        array[back] = item;
        size++;
        return true;
    }

    public T poll(){
        if(size == 0)   return null;

        front = (front + 1) % array.length;
        T item = (T) array[front];
        array[front] = null;
        size--;

        if(array.length > DEFAULT_CAPACITY && size < array.length/4)
            resize(Math.max(DEFAULT_CAPACITY, array.length/2));
        return item;
    }
    public T reomve(){
        T item = poll();
        if(item == null)    throw new NoSuchElementException();
        return item;
    }

    public T peek(){
        if(size == 0)   return null;
        T item = (T)array[(front+1)%array.length];
        return item;
    }

    public boolean isEmpty(){
        return size == 0;
    }

    public void clear(){
        Arrays.fill(array, null);
        front = back = size = 0;
    }
}
