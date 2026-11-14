
import pandas as pd
import random

# Reproducible dataset generation
random.seed(42)

# Each category has a known root cause and realistic incident examples.
root_causes = {
    "VPN Configuration Error": {
        "category": "Network",
        "assignment_group": "Network Support",
        "shorts": [
            "VPN connection fails after configuration change",
            "Remote users cannot connect to VPN",
            "VPN authentication succeeds but connection drops",
            "Employees report VPN access errors",
            "VPN tunnel fails to establish",
            "Remote access unavailable after maintenance",
            "VPN client reports a configuration mismatch",
            "Intermittent disconnections affect remote workers",
        ],
        "descriptions": [
            "Remote employees cannot access internal applications because the VPN tunnel fails to establish.",
            "The VPN client reports a configuration mismatch following a recent network change.",
            "Users authenticate successfully but the VPN connection drops before internal resources load.",
            "Remote access stopped working after VPN gateway settings were updated.",
            "Employees experience repeated VPN disconnections when connecting from outside the office.",
            "The VPN profile contains outdated connection settings and prevents access to company resources.",
            "Several users receive VPN connection errors after a configuration update.",
            "The remote access tunnel fails because the client and gateway settings do not match.",
        ],
        "resolution": "Review and correct the VPN profile and gateway configuration."
    },

    "Database Connectivity Failure": {
        "category": "Database",
        "assignment_group": "Database Support",
        "shorts": [
            "Application cannot connect to database",
            "Database connection timeout reported",
            "Transactions fail due to database connectivity",
            "Database service connection unavailable",
            "Application reports database connection refused",
            "Database queries fail intermittently",
            "Connection pool cannot reach database server",
            "Database requests time out during processing",
        ],
        "descriptions": [
            "The application cannot establish a connection to the database server and requests time out.",
            "Database connections are refused when the application attempts to process transactions.",
            "Users cannot load records because the database endpoint is unreachable.",
            "The connection pool repeatedly fails to acquire a database connection.",
            "Database queries fail because the application loses its connection to the database host.",
            "Transaction processing stops when database connection attempts exceed the timeout limit.",
            "The application logs connection errors while attempting to reach the configured database server.",
            "Multiple requests fail because the database service cannot be reached from the application server.",
        ],
        "resolution": "Restore database connectivity and verify the connection settings and service availability."
    },

    "Access Permission Misconfiguration": {
        "category": "Access",
        "assignment_group": "Identity and Access Support",
        "shorts": [
            "User cannot access restricted application",
            "Permission denied when opening business application",
            "Employee unable to access assigned resources",
            "Access denied after role change",
            "User lacks required application permissions",
            "Document access blocked by permission settings",
            "Role assignment prevents application access",
            "Employee receives an authorization error",
        ],
        "descriptions": [
            "The user receives an authorization error when opening an application they need for work.",
            "An employee cannot access a business resource because the required role is missing.",
            "The application denies access after the user's permissions were changed.",
            "A user can sign in successfully but cannot open a restricted business document.",
            "The assigned account lacks the permissions required to complete an application workflow.",
            "Access is blocked because the account has an incorrect role assignment.",
            "The user receives a permission denied message when accessing an internal tool.",
            "A role update has left the employee without the required application privileges.",
        ],
        "resolution": "Verify the user's role and grant the appropriate authorized permissions."
    },

    "DNS Resolution Failure": {
        "category": "Network",
        "assignment_group": "Network Support",
        "shorts": [
            "Internal hostname cannot be resolved",
            "DNS lookup fails for business application",
            "Users cannot reach service by hostname",
            "Name resolution errors across office network",
            "Application hostname resolves incorrectly",
            "DNS requests time out",
            "Internal service name returns no address",
            "Hostname access fails while IP access works",
        ],
        "descriptions": [
            "Users cannot open an internal application because its hostname fails to resolve.",
            "DNS queries time out when clients attempt to locate an internal service.",
            "The application is reachable by IP address but not by its configured hostname.",
            "Employees receive name resolution errors when accessing company websites.",
            "A service hostname returns an incorrect address following a DNS record change.",
            "Internal hostnames cannot be resolved by the configured DNS server.",
            "Requests fail because the DNS resolver cannot return the address of the target service.",
            "Multiple clients experience hostname lookup failures on the office network.",
        ],
        "resolution": "Check DNS records, resolver configuration and name resolution."
    },

    "Disk Space Exhaustion": {
        "category": "Storage",
        "assignment_group": "Infrastructure Support",
        "shorts": [
            "Server disk is almost full",
            "Application fails because storage is exhausted",
            "Disk space alert triggered on server",
            "Logs cannot be written due to full disk",
            "Server reports insufficient storage",
            "Application files cannot be saved",
            "Disk usage reaches critical threshold",
            "Temporary files fill system drive",
        ],
        "descriptions": [
            "The server cannot write application logs because the system disk has run out of space.",
            "A service fails to start after available disk storage reaches zero.",
            "The application cannot save uploaded files because the storage volume is full.",
            "Disk usage has reached a critical level due to accumulated log files.",
            "The server reports insufficient disk space while processing new requests.",
            "Temporary files have consumed the available capacity on the system drive.",
            "Database backups fail because the destination storage volume has no free space.",
            "Applications report write errors because the server storage partition is exhausted.",
        ],
        "resolution": "Free unnecessary storage, review log retention and expand capacity if required."
    },

    "Memory Resource Exhaustion": {
        "category": "Performance",
        "assignment_group": "Infrastructure Support",
        "shorts": [
            "Server memory usage reaches critical level",
            "Application slows due to high RAM usage",
            "Service crashes during memory pressure",
            "Out of memory error on application server",
            "Memory consumption increases continuously",
            "Server becomes unresponsive under memory load",
            "Application process is terminated for memory use",
            "High RAM utilization causes request failures",
        ],
        "descriptions": [
            "The application becomes unresponsive when server memory usage reaches a critical level.",
            "An out of memory error causes the application process to terminate unexpectedly.",
            "Memory consumption increases continuously until the service crashes.",
            "Users experience slow responses when available RAM becomes very low.",
            "The operating system terminates a process because the server runs out of memory.",
            "Requests begin failing when application memory usage exceeds available capacity.",
            "The server becomes unstable during periods of high memory utilization.",
            "A service repeatedly restarts after memory pressure exhausts available RAM.",
        ],
        "resolution": "Investigate memory usage, identify excessive processes and optimize resource allocation."
    },

    "Application Service Failure": {
        "category": "Application",
        "assignment_group": "Application Support",
        "shorts": [
            "Business application service is unavailable",
            "Application service fails to start",
            "Users receive internal server errors",
            "Application process stops unexpectedly",
            "Service restart required after application failure",
            "Application endpoint returns server error",
            "Business workflow unavailable due to service crash",
            "Application service repeatedly stops",
        ],
        "descriptions": [
            "Users receive server errors because the business application service is not running.",
            "The application process exits unexpectedly and the service endpoint becomes unavailable.",
            "A business workflow fails after the application service stops responding.",
            "The application cannot start because its service process exits during startup.",
            "Repeated service failures prevent users from accessing the business application.",
            "The application endpoint returns server errors while its backend service is unavailable.",
            "Users cannot complete transactions because the application service has crashed.",
            "The application service stops repeatedly and requires investigation of its runtime logs.",
        ],
        "resolution": "Inspect application logs, identify the service failure and restore normal operation."
    },

    "Expired SSL Certificate": {
        "category": "Security",
        "assignment_group": "Security Support",
        "shorts": [
            "Website certificate has expired",
            "Browser reports an SSL certificate warning",
            "HTTPS connection rejected due to certificate validity",
            "Secure application access fails certificate check",
            "TLS handshake fails after certificate expiry",
            "Users receive certificate expiration alerts",
            "Secure endpoint presents an invalid certificate",
            "Application reports certificate validation failure",
        ],
        "descriptions": [
            "Browsers warn users that the website certificate has expired.",
            "The HTTPS connection is rejected because the server certificate is outside its validity period.",
            "The TLS handshake fails when clients validate the expired certificate.",
            "Users cannot securely access the application because certificate validation fails.",
            "The server presents an expired certificate during a secure connection attempt.",
            "A certificate expiration alert appears when employees open the business portal.",
            "The secure endpoint fails verification because its certificate is no longer valid.",
            "Clients reject HTTPS requests after the server certificate expires.",
        ],
        "resolution": "Renew and install the valid certificate, then verify the secure connection."
    },

    "Email Server Configuration Error": {
        "category": "Email",
        "assignment_group": "Messaging Support",
        "shorts": [
            "Outgoing emails fail to send",
            "Mail server rejects outgoing messages",
            "Email delivery fails after settings change",
            "SMTP connection cannot be established",
            "Business emails remain stuck in the outbox",
            "Mail relay authentication fails",
            "Email client cannot reach configured mail server",
            "Messages bounce due to mail configuration",
        ],
        "descriptions": [
            "Outgoing messages fail because the email client cannot connect to the configured SMTP server.",
            "The mail relay rejects messages after an email configuration change.",
            "Business emails remain in the outbox because SMTP authentication fails.",
            "The email service cannot establish a connection to its configured mail server.",
            "Messages bounce because the outgoing mail server settings are incorrect.",
            "Email delivery stops after the SMTP host and port settings are changed.",
            "Users cannot send email because the mail relay configuration is invalid.",
            "The email client reports a server connection error when sending messages.",
        ],
        "resolution": "Verify SMTP host, port, authentication and mail relay configuration."
    },

    "Software Update Compatibility Issue": {
        "category": "Software",
        "assignment_group": "Desktop Support",
        "shorts": [
            "Application fails after software update",
            "New update causes application compatibility error",
            "Desktop tool crashes after version upgrade",
            "Application plugin incompatible with updated software",
            "Users report errors after workstation patch",
            "Business tool fails following version change",
            "Updated application cannot load required component",
            "Software upgrade causes repeated application errors",
        ],
        "descriptions": [
            "The application crashes after a software update changes a required component.",
            "A plugin fails to load because it is incompatible with the newly installed application version.",
            "Users experience application errors after a workstation patch was applied.",
            "The business tool cannot start following a version upgrade.",
            "A required component is no longer supported by the updated software release.",
            "Repeated crashes began immediately after the desktop application was upgraded.",
            "The updated application fails to load an extension that worked in the previous version.",
            "Users report compatibility errors after the latest software release was installed.",
        ],
        "resolution": "Verify software compatibility and apply a supported update or rollback if needed."
    },
}

urgencies = ["1 - High", "2 - Medium", "3 - Low"]
impacts = ["1 - High", "2 - Medium", "3 - Low"]

context_phrases = [
    "The issue affects one employee.",
    "Several users have reported the same issue.",
    "The issue was reported during normal business hours.",
    "The problem began after a recent system change.",
    "The support team has received repeated reports.",
]

records = []
incident_number = 1
examples_per_cause = 50

for cause, info in root_causes.items():
    for i in range(examples_per_cause):
        template_index = i % len(info["descriptions"])

        short_description = info["shorts"][template_index]
        description = info["descriptions"][template_index]

        # Add controlled wording variation without changing the known label.
        if i % 3 == 1:
            description = "Reported incident: " + description
        elif i % 3 == 2:
            description = description + " " + random.choice(context_phrases)

        urgency = random.choices(
            urgencies, weights=[0.25, 0.50, 0.25], k=1
        )[0]
        impact = random.choices(
            impacts, weights=[0.20, 0.55, 0.25], k=1
        )[0]

        records.append({
            "incident_id": f"INC{incident_number:05d}",
            "short_description": short_description,
            "description": description,
            "urgency": urgency,
            "impact": impact,
            "category": info["category"],
            "assignment_group": info["assignment_group"],
            "root_cause": cause,
            "resolution_note": info["resolution"],
        })

        incident_number += 1

# Shuffle the records so that causes are not grouped together.
random.shuffle(records)

df = pd.DataFrame(records)
df.to_csv("incidents.csv", index=False, encoding="utf-8")

print("Dataset generated successfully!")
print(f"Total incidents: {len(df)}")
print(f"Root-cause categories: {df['root_cause'].nunique()}")
print("\nIncidents per root cause:")
print(df["root_cause"].value_counts().to_string())

print("\nFirst five records:")
print(df[["incident_id", "short_description", "root_cause"]].head().to_string(index=False))

print("\nSaved to incidents.csv in the current project folder.")
