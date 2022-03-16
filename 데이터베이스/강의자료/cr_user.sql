create user test01 identified by test01
default tablespace users
temporary tablespace temp;

grant connect,resource,create view to test01;