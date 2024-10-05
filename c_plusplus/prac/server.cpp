#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

int main() {
    // Create the server socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create server socket\n";
        return 1;
    }

    // Define the server address
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;  // Bind to all available interfaces
    address.sin_port = htons(8080);        // Use port 8080

    // Bind the socket to the specified address and port
    if (bind(server_fd, (struct sockaddr*)&address, addrlen) < 0) {
        std::cerr << "Bind failed\n";
        close(server_fd);
        return 1;
    }

    return 0;
}