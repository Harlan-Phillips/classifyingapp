flowchart TD
    munchDB[Munch DB]
    culinaDB[Culina DB]
    managerTable[Managers Table (Location IDs + Chain)]
    googleAuth[Google OAuth Login]
    dashboard[Inventory Dashboard]
    checkManager{Is user a manager?}
    accessGranted[Grant Access to Dashboard]
    accessDenied[Show Access Denied Message]
    filterEmployees[Filter Employees (by Location & Chain)]
    viewInvite[View / Invite Employees]

    munchDB --> culinaDB
    culinaDB --> managerTable
    googleAuth --> dashboard
    managerTable --> dashboard
    dashboard --> checkManager
    checkManager -->|Yes| accessGranted
    checkManager -->|No| accessDenied
    accessGranted --> filterEmployees
    filterEmployees --> viewInvite
