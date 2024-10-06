import logging

import openstack

logger = logging.getLogger(__name__)

VOLUME_WAIT_TIME = 300
SERVER_WAIT_TIME = 300


def get_connection(auth_url, username, password, project_name,
                   user_domain_name, project_domain_name):
    return openstack.connect(
        auth_url=auth_url,
        project_name=project_name,
        username=username,
        password=password,
        user_domain_name=user_domain_name,
        project_domain_name=project_domain_name,
    )


def get_floating_ip(conn, floating_network_name):
    network = conn.network.find_network(floating_network_name)

    project_name = conn.auth["project_name"]
    project = conn.identity.find_project(project_name)

    for ip in conn.network.ips(project_id=project.id,
                               floating_network_id=network.id,
                               status="DOWN", fixed_ip_address=None):
        logger.info(f"Found available floating IP: {ip.floating_ip_address}")
        return ip

    logger.info("No available floating IP found. Creating a new floating IP "
                f"in network: {floating_network_name}")
    floating_network = conn.network.find_network(floating_network_name)
    if not floating_network:
        raise ValueError(
            f"Floating network '{floating_network_name}' not found.")

    new_ip = conn.network.create_ip(floating_network_id=floating_network.id)
    logger.info(f"Created new floating IP: {new_ip.floating_ip_address}")
    return new_ip


def create_server(conn, instance_name, image_name, flavor_name, keypair_name,
                  network_name, volume_size_gb):
    image = conn.compute.find_image(image_name)
    if not image:
        raise ValueError(f"Image \"{image_name}\" not found")

    flavor = conn.compute.find_flavor(flavor_name)
    if not flavor:
        raise ValueError(f"Flavor \"{flavor_name}\" not found")

    network = conn.network.find_network(network_name)
    if not network:
        raise ValueError(f"Network \"{network_name}\" not found")

    volume_name = f"{instance_name}-boot-volume"
    logger.info(f"Creating volume \"{volume_name}\" "
                f"with size {volume_size_gb} GB")

    volume = conn.block_storage.create_volume(
        size=volume_size_gb,
        name=volume_name,
        image_id=image.id
    )

    logger.info(f"Waiting for volume \"{volume_name}\"")
    volume = conn.block_storage.wait_for_status(
        volume, status='available', failures=['error'], interval=2,
        wait=VOLUME_WAIT_TIME)

    block_device_mapping = [{
        'boot_index': 0,
        'uuid': volume.id,
        'source_type': 'volume',
        'destination_type': 'volume',
        'delete_on_termination': True
    }]

    logger.info(f"Creating server \"{instance_name}\"")
    server = conn.compute.create_server(
        name=instance_name,
        image_id=image.id,
        flavor_id=flavor.id,
        networks=[{"uuid": network.id}],
        key_name=keypair_name,
        block_device_mapping_v2=block_device_mapping
    )

    logger.info(f"Waiting for server \"{server.name}\" "
                f"with id: \"{server.id}\"")
    return conn.compute.wait_for_server(server, wait=SERVER_WAIT_TIME)


def attach_floating_ip(conn, server_id, floating_ip):
    server = conn.compute.find_server(server_id)

    server_port = list(conn.network.ports(device_id=server.id))[0]
    logger.info(f"Attaching floating ip {floating_ip.floating_ip_address} "
                f"to server \"{server.id}\"")
    conn.network.update_ip(floating_ip, port_id=server_port.id)

    return floating_ip.floating_ip_address


def delete_server(conn, server_id):
    server = conn.compute.find_server(server_id)
    logger.info(f"Deleting server \"{server.id}\"")
    conn.compute.delete_server(server)
